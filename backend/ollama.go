package backend

import (
	"context"
	"encoding/json"
	"net/http"

	"github.com/tamnd/local-llm/config"
)

// Ollama adapts the Ollama runtime (doc 06, doc 08 section 5.2). Loading and
// unloading are done through the keep-alive field on a generate call rather than
// a dedicated endpoint: a request with keep_alive set loads the model, and one
// with keep_alive zero tells Ollama to evict it.
type Ollama struct {
	proxy *proxy
}

// ID returns the backend id "ollama".
func (o *Ollama) ID() string { return config.BackendOllama }

// Load issues a warm-up generate with an empty prompt so the model is resident
// before the first real request. The keep_alive comes from the model's params,
// defaulting to "30m".
func (o *Ollama) Load(ctx context.Context, entry config.ModelEntry) error {
	keepAlive := paramString(entry.Params, "keep_alive", "30m")
	body, _ := json.Marshal(map[string]any{
		"model":      entry.UpstreamModel,
		"prompt":     "",
		"keep_alive": keepAlive,
		"stream":     false,
	})
	return o.proxy.postJSON(ctx, entry.BaseURL, "/api/generate", body, "")
}

// Unload sends keep_alive zero, the documented heartbeat trick for telling
// Ollama to release the model's VRAM.
func (o *Ollama) Unload(ctx context.Context, entry config.ModelEntry) error {
	body, _ := json.Marshal(map[string]any{
		"model":      entry.UpstreamModel,
		"prompt":     "",
		"keep_alive": 0,
		"stream":     false,
	})
	return o.proxy.postJSON(ctx, entry.BaseURL, "/api/generate", body, "")
}

// Forward proxies the request to Ollama, normalizing the SSE or JSON response.
func (o *Ollama) Forward(ctx context.Context, entry config.ModelEntry, req *Request, w http.ResponseWriter) (*Result, error) {
	return o.proxy.forward(ctx, o.ID(), entry.BaseURL, req, w)
}

// Healthy probes Ollama's /api/tags, which responds once the daemon is up.
func (o *Ollama) Healthy(ctx context.Context, entry config.ModelEntry) error {
	return o.proxy.healthCheck(ctx, entry.BaseURL, "/api/tags")
}

// paramString reads a string param from a model entry's params map, returning
// def when the key is absent or not a string. The YAML decoder gives us
// map[string]any, so a small typed accessor keeps the adapters readable.
func paramString(params map[string]any, key, def string) string {
	if params == nil {
		return def
	}
	if v, ok := params[key].(string); ok && v != "" {
		return v
	}
	return def
}
