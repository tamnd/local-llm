package backend

import (
	"context"
	"net/http"
	"net/url"
	"strconv"
	"time"

	"github.com/tamnd/local-llm/config"
)

// Llama adapts llama-server (llama.cpp). Unlike Ollama and TabbyAPI it has no
// runtime model swap: the running process is the resident model (doc 08 section
// 5.2). Loading a different model means stopping the process and starting a new
// one with different flags, which the procTable handles.
type Llama struct {
	proxy *proxy
	procs *procTable
}

// ID returns the backend id "llama".
func (l *Llama) ID() string { return config.BackendLlama }

// Load starts llama-server for entry.UpstreamModel (a GGUF path) and waits for
// its /health endpoint before returning. The binary path comes from the model's
// "bin" param; the rest of the flags are assembled from params with the
// doc 14 section 2.2 defaults.
func (l *Llama) Load(ctx context.Context, entry config.ModelEntry) error {
	// Adopt an already-serving llama-server instead of spawning one. The process
	// is the resident model, so a healthy endpoint at the configured port is the
	// model this entry wants. This is the path used when llama-server runs as an
	// externally managed service and, on the RTX 4090 box, when it runs in WSL2
	// while the gateway runs on the Windows host and so cannot exec the Linux
	// binary itself: the gateway reaches the WSL server over localhostForwarding
	// and just proxies to it. Same reasoning as VLLM.Load.
	if l.Healthy(ctx, entry) == nil {
		return nil
	}
	bin := paramString(entry.Params, "bin", "")
	args := buildLlamaArgs(entry)
	ready := func(c context.Context) error { return l.Healthy(c, entry) }
	return l.procs.swap(ctx, entry.UpstreamModel, bin, args, func(c context.Context) error {
		return waitHealthy(c, 60*time.Second, ready)
	})
}

// Unload stops the llama-server process, which frees its VRAM.
func (l *Llama) Unload(ctx context.Context, _ config.ModelEntry) error {
	return l.procs.stop(ctx)
}

// Forward proxies the request to the running llama-server, normalizing output.
func (l *Llama) Forward(ctx context.Context, entry config.ModelEntry, req *Request, w http.ResponseWriter) (*Result, error) {
	return l.proxy.forward(ctx, l.ID(), entry.BaseURL, req, w)
}

// Healthy probes llama-server's /health endpoint.
func (l *Llama) Healthy(ctx context.Context, entry config.ModelEntry) error {
	return l.proxy.healthCheck(ctx, entry.BaseURL, "/health")
}

// buildLlamaArgs assembles the llama-server command line from a model entry. The
// flags mirror the box defaults in doc 14 section 2.2: all layers on GPU, flash
// attention on, q8 KV cache, the configured context size, and the port parsed
// from the base URL.
func buildLlamaArgs(entry config.ModelEntry) []string {
	args := []string{
		"--model", entry.UpstreamModel,
		"--host", "127.0.0.1",
		"--n-gpu-layers", strconv.Itoa(paramInt(entry.Params, "n_gpu_layers", 999)),
		"--ctx-size", strconv.Itoa(paramInt(entry.Params, "n_ctx", 32768)),
	}
	if port := portFromURL(entry.BaseURL); port != "" {
		args = append(args, "--port", port)
	}
	if paramBool(entry.Params, "flash_attn", true) {
		args = append(args, "--flash-attn")
	}
	args = append(args,
		"--cache-type-k", paramString(entry.Params, "cache_type_k", "q8_0"),
		"--cache-type-v", paramString(entry.Params, "cache_type_v", "q8_0"),
	)
	// Vision. A multimodal GGUF carries its projector in a separate file, and
	// llama-server serves the model text-only when --mmproj is absent: the image
	// parts of a request are dropped and the completion comes back describing
	// nothing, with no error anywhere. Emitted only when configured, so the
	// text-only entries keep the exact flag set they had.
	if mmproj := paramString(entry.Params, "mmproj_path", ""); mmproj != "" {
		args = append(args, "--mmproj", mmproj)
	}
	// Sampling. llama-server's built-in defaults are not every model's
	// recommended ones (Muse Glimmer wants temp 1.0 / top-p 0.95 / top-k 64 per
	// Unsloth's model card), and a mismatch here shows up as quality drift
	// rather than an error, so each is emitted only when a model pins it.
	for _, s := range []struct {
		key, flag string
	}{
		{"temp", "--temp"},
		{"top_p", "--top-p"},
		{"top_k", "--top-k"},
	} {
		if v := paramNumString(entry.Params, s.key); v != "" {
			args = append(args, s.flag, v)
		}
	}
	// Speculative decoding. A vocab-matched draft GGUF (for Muse Glimmer, Meta's
	// DFlash head) lets llama-server verify several drafted tokens per target
	// forward pass, which is a decode-side win on memory-bandwidth-bound models
	// and a no-op when the draft mispredicts. Only emitted when a draft is
	// configured, so single-model entries keep the exact flag set they had.
	// The flag names are the current spelling: --draft-max and --draft-min were
	// removed upstream in favour of --spec-draft-n-max and --spec-draft-n-min,
	// and llama-server exits on the old ones rather than warning.
	if draft := paramString(entry.Params, "draft_path", ""); draft != "" {
		args = append(args,
			"--spec-draft-model", draft,
			"--spec-draft-ngl", strconv.Itoa(paramInt(entry.Params, "gpu_layers_draft", 99)),
			"--spec-draft-n-max", strconv.Itoa(paramInt(entry.Params, "draft_max", 6)),
			"--spec-draft-n-min", strconv.Itoa(paramInt(entry.Params, "draft_min", 1)),
		)
	}
	return args
}

// portFromURL extracts the port from an http://host:port base URL, or "" if the
// URL has no explicit port.
func portFromURL(base string) string {
	u, err := url.Parse(base)
	if err != nil {
		return ""
	}
	return u.Port()
}

// paramNumString renders a numeric knob as the string its flag takes, and ""
// when the key is absent. It accepts the number unquoted as well as quoted:
// YAML decodes `temp: 1.0` to a float64, which paramString would drop on the
// floor, and a sampling flag that silently goes missing is the kind of bug that
// only shows up as worse output weeks later.
func paramNumString(params map[string]any, key string) string {
	if params == nil {
		return ""
	}
	switch v := params[key].(type) {
	case string:
		return v
	case int:
		return strconv.Itoa(v)
	case int64:
		return strconv.FormatInt(v, 10)
	case float64:
		return strconv.FormatFloat(v, 'g', -1, 64)
	}
	return ""
}

func paramInt(params map[string]any, key string, def int) int {
	if params == nil {
		return def
	}
	switch v := params[key].(type) {
	case int:
		return v
	case int64:
		return int(v)
	case float64:
		return int(v)
	case string:
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func paramBool(params map[string]any, key string, def bool) bool {
	if params == nil {
		return def
	}
	if v, ok := params[key].(bool); ok {
		return v
	}
	return def
}
