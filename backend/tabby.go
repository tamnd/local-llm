package backend

import (
	"context"
	"encoding/json"
	"net/http"

	"github.com/tamnd/local-llm/config"
)

// Tabby adapts TabbyAPI (ExLlamaV3). It is the cleanest backend for hot-swap
// because it was designed for it: dedicated load and unload endpoints that
// return once VRAM is allocated or freed (doc 08 section 5.2). The load and
// unload calls carry the admin token TabbyAPI requires.
type Tabby struct {
	proxy *proxy
}

// ID returns the backend id "tabby".
func (t *Tabby) ID() string { return config.BackendTabby }

// Load posts to /v1/model/load with the model name and any per-model params
// (max_seq_len, cache_mode, chunk_size). TabbyAPI responds 200 once resident.
func (t *Tabby) Load(ctx context.Context, entry config.ModelEntry) error {
	payload := map[string]any{"model_name": entry.UpstreamModel}
	for k, v := range entry.Params {
		if k == "admin_key" {
			continue
		}
		payload[k] = v
	}
	body, _ := json.Marshal(payload)
	return t.proxy.postJSON(ctx, entry.BaseURL, "/v1/model/load", body, t.adminKey(entry))
}

// Unload posts to /v1/model/unload; returns once VRAM is free.
func (t *Tabby) Unload(ctx context.Context, entry config.ModelEntry) error {
	return t.proxy.postJSON(ctx, entry.BaseURL, "/v1/model/unload", []byte("{}"), t.adminKey(entry))
}

// Forward proxies the request to TabbyAPI, normalizing the response.
func (t *Tabby) Forward(ctx context.Context, entry config.ModelEntry, req *Request, w http.ResponseWriter) (*Result, error) {
	return t.proxy.forward(ctx, t.ID(), entry.BaseURL, req, w)
}

// Healthy probes TabbyAPI's /v1/models.
func (t *Tabby) Healthy(ctx context.Context, entry config.ModelEntry) error {
	return t.proxy.healthCheck(ctx, entry.BaseURL, "/v1/models")
}

func (t *Tabby) adminKey(entry config.ModelEntry) string {
	return paramString(entry.Params, "admin_key", "")
}
