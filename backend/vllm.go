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
	if q := paramString(entry.Params, "quantization", "awq"); q != "" {
		args = append(args, "--quantization", q)
	}
	return args
}
