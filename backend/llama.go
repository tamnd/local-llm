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
