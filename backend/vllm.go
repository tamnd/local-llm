package backend

import (
	"context"
	"net/http"
	"strconv"
	"time"

	"github.com/tamnd/local-llm/config"
)

// VLLM adapts vLLM, the concurrent-serving runtime used for the M6 experiment
// (doc 06 section 7, doc 08 section 5.2). Like llama-server it is process-driven
// and pre-allocates most of VRAM on startup, so it is never the default swap
// target; it is loaded only when a client explicitly targets a vLLM-hosted
// model. Startup is slow (weight load plus KV pool allocation), so the readiness
// gate allows a generous timeout.
type VLLM struct {
	proxy *proxy
	procs *procTable
}

// ID returns the backend id "vllm".
func (v *VLLM) ID() string { return config.BackendVLLM }

// Load starts `vllm serve <model>` with the box's experiment flags
// (doc 14 section 2.6) and waits for the server to answer health checks before
// returning. vLLM's KV-pool allocation and CUDA graph work can take minutes, so
// the readiness timeout is 300 seconds.
func (v *VLLM) Load(ctx context.Context, entry config.ModelEntry) error {
	// Adopt an already-serving vLLM instead of spawning one. vLLM pins a single
	// model per server, so a healthy endpoint at the configured port is the model
	// this entry wants. This is the path used when vLLM runs as an externally
	// managed service (a systemd unit) and, on the RTX 4090 box, when it runs in
	// WSL2 while the gateway runs on the Windows host and so cannot exec the Linux
	// `vllm` binary itself: the gateway reaches the WSL server over
	// localhostForwarding and just proxies to it.
	if v.Healthy(ctx, entry) == nil {
		return nil
	}
	bin := paramString(entry.Params, "bin", "vllm")
	args := buildVLLMArgs(entry)
	ready := func(c context.Context) error { return v.Healthy(c, entry) }
	return v.procs.swap(ctx, entry.UpstreamModel, bin, args, func(c context.Context) error {
		return waitHealthy(c, 300*time.Second, ready)
	})
}

// Unload stops the vLLM process, releasing its weights and KV pool.
func (v *VLLM) Unload(ctx context.Context, _ config.ModelEntry) error {
	return v.procs.stop(ctx)
}

// Forward proxies the request to the running vLLM server, normalizing output.
func (v *VLLM) Forward(ctx context.Context, entry config.ModelEntry, req *Request, w http.ResponseWriter) (*Result, error) {
	return v.proxy.forward(ctx, v.ID(), entry.BaseURL, req, w)
}

// Healthy probes vLLM's /health endpoint.
func (v *VLLM) Healthy(ctx context.Context, entry config.ModelEntry) error {
	return v.proxy.healthCheck(ctx, entry.BaseURL, "/health")
}

// buildVLLMArgs assembles the vLLM serve command line, constraining GPU memory
// so the desktop compositor keeps a slice (doc 06 section 7) and forcing eager
// mode because CUDA graph capture is unreliable on the Windows WDDM path.
func buildVLLMArgs(entry config.ModelEntry) []string {
	args := []string{
		"serve", entry.UpstreamModel,
		"--host", "127.0.0.1",
		"--gpu-memory-utilization", paramString(entry.Params, "gpu_memory_utilization", "0.85"),
		"--max-model-len", strconv.Itoa(paramInt(entry.Params, "max_model_len", 32768)),
		"--tensor-parallel-size", "1",
	}
	if port := portFromURL(entry.BaseURL); port != "" {
		args = append(args, "--port", port)
	}
	if paramBool(entry.Params, "enforce_eager", true) {
		args = append(args, "--enforce-eager")
	}
	// quantization defaults to empty, not "awq". It used to default to "awq" from
	// when every vLLM entry on the roster was an AWQ checkout, which meant any
	// entry that did not set the key got --quantization awq appended to it. That
	// is wrong for a checkpoint whose weights are already in their served format:
	// both remaining entries here (Qwen3.8-27B at FP8, gpt-oss-20b at MXFP4) carry
	// the scheme in config.json and vLLM picks it up on its own, while an explicit
	// --quantization awq makes it try to load fp8 tensors through the AWQ path and
	// fail at startup. The bug never fired because the box runs vLLM from systemd
	// units and the gateway adopts the healthy port instead of spawning, so this
	// function's output was never exercised on those entries.
	if q := paramString(entry.Params, "quantization", ""); q != "" {
		args = append(args, "--quantization", q)
	}
	// An fp8 KV cache halves the KV footprint, which is what lets a 4-bit 32B
	// hold a 32k context on the 24 GB card (a 32k fp16 cache is ~8 GiB and
	// overflows). Only passed when the model sets it, so fp16-cache models are
	// unaffected.
	if kv := paramString(entry.Params, "kv_cache_dtype", ""); kv != "" {
		args = append(args, "--kv-cache-dtype", kv)
	}
	// Spill this many GiB of weights to host RAM. The only reason to reach for it
	// is a checkpoint that cannot fit at all -- Qwen3.8-27B-FP8 is 28.77 GiB
	// against 23.99 GiB of card -- and it costs a PCIe round trip per offloaded
	// layer per forward pass, so it is a correctness knob rather than a tuning
	// one. Absent by default; the offloaded weights are pinned in host RAM, so
	// the host needs the headroom before this is set.
	if off := paramString(entry.Params, "cpu_offload_gb", ""); off != "" {
		args = append(args, "--cpu-offload-gb", off)
	}
	// Chunked prefill was already a documented key on three model entries and was
	// silently dropped here, so the flag never reached vLLM. Accepted as a bool or
	// as the quoted "true" the existing entries write.
	if paramBoolLoose(entry.Params, "enable_chunked_prefill", false) {
		args = append(args, "--enable-chunked-prefill")
	}
	return args
}

// paramBoolLoose reads a boolean knob that config may express either as a YAML
// bool or as a quoted string. The vLLM entries quote their numeric and boolean
// params ("0.92", "true") while the llama entries do not, and a knob that only
// understands one spelling goes missing without an error.
func paramBoolLoose(params map[string]any, key string, def bool) bool {
	if params == nil {
		return def
	}
	switch v := params[key].(type) {
	case bool:
		return v
	case string:
		if b, err := strconv.ParseBool(v); err == nil {
			return b
		}
	}
	return def
}
