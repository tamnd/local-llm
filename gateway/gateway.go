// Package gateway is the HTTP front door. It exposes the OpenAI-compatible
// surface (GET /v1/models, POST /v1/chat/completions, /v1/completions,
// /v1/embeddings) plus a small admin plane (GET /healthz, GET /admin/status,
// POST /admin/load, POST /admin/unload). Every data-plane request flows through
// the same pipeline: authenticate, resolve the model, make it resident via the
// manager, forward to the backend, then log. The gateway holds no model state of
// its own; residency lives in the manager and routing in the router.
package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"time"

	"github.com/tamnd/local-llm/auth"
	"github.com/tamnd/local-llm/backend"
	"github.com/tamnd/local-llm/config"
	"github.com/tamnd/local-llm/manager"
	"github.com/tamnd/local-llm/observe"
	"github.com/tamnd/local-llm/router"
)

// maxBodyBytes caps a request body. Chat payloads with long contexts are large
// but bounded; this stops a runaway client from exhausting memory.
const maxBodyBytes = 32 << 20 // 32 MiB

// Gateway wires the request pipeline together. Build it with New and serve its
// Handler.
type Gateway struct {
	router  *router.Router
	mgr     *manager.Manager
	reg     backend.Registry
	auth    *auth.Authenticator
	admin   *auth.Authenticator
	log     *observe.Logger
	version string
}

// New constructs a Gateway from the validated config and its collaborators.
func New(cfg *config.Config, rt *router.Router, mgr *manager.Manager, reg backend.Registry, log *observe.Logger, version string) *Gateway {
	var adminAuth *auth.Authenticator
	if cfg.Auth.AdminToken != "" {
		adminAuth = auth.New([]string{cfg.Auth.AdminToken})
	}
	return &Gateway{
		router:  rt,
		mgr:     mgr,
		reg:     reg,
		auth:    auth.New(cfg.Auth.Tokens),
		admin:   adminAuth,
		log:     log,
		version: version,
	}
}

// Handler returns the HTTP mux serving every route.
func (g *Gateway) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", g.handleHealth)
	mux.HandleFunc("GET /v1/models", g.handleModels)
	mux.HandleFunc("POST /v1/chat/completions", g.handleInference("/v1/chat/completions"))
	mux.HandleFunc("POST /v1/completions", g.handleInference("/v1/completions"))
	mux.HandleFunc("POST /v1/embeddings", g.handleInference("/v1/embeddings"))
	mux.HandleFunc("GET /admin/status", g.handleStatus)
	mux.HandleFunc("POST /admin/load", g.handleAdminLoad)
	mux.HandleFunc("POST /admin/unload", g.handleAdminUnload)
	return mux
}

// handleHealth is an unauthenticated liveness probe. It reports the active model
// and residency state so the tailnet box can be checked without a token.
func (g *Gateway) handleHealth(w http.ResponseWriter, _ *http.Request) {
	st := g.mgr.Status()
	writeJSON(w, http.StatusOK, map[string]any{
		"status":       "ok",
		"version":      g.version,
		"active_model": st.ActiveModel,
		"state":        st.State,
	})
}

// handleModels returns the model list in OpenAI's shape. Aliases are not listed;
// clients see only canonical model ids.
func (g *Gateway) handleModels(w http.ResponseWriter, r *http.Request) {
	if _, err := g.authenticate(r); err != nil {
		writeAuthError(w, err)
		return
	}
	ids := g.router.IDs()
	data := make([]map[string]any, 0, len(ids))
	for _, id := range ids {
		data = append(data, map[string]any{
			"id":       id,
			"object":   "model",
			"owned_by": "local-llm",
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": data})
}

// handleInference returns the shared handler for the three inference endpoints.
// They differ only in path; the pipeline is identical.
func (g *Gateway) handleInference(path string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		token, autherr := g.authenticate(r)
		if autherr != nil {
			writeAuthError(w, autherr)
			return
		}

		body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxBodyBytes))
		if err != nil {
			writeError(w, http.StatusRequestEntityTooLarge, "request_too_large", "request body exceeds limit")
			return
		}

		var probe struct {
			Model  string `json:"model"`
			Stream bool   `json:"stream"`
		}
		if err := json.Unmarshal(body, &probe); err != nil {
			writeError(w, http.StatusBadRequest, "invalid_json", "request body is not valid JSON")
			return
		}

		resolved, ok := g.router.Resolve(probe.Model)
		if !ok {
			writeError(w, http.StatusNotFound, "model_not_found", "unknown model: "+probe.Model)
			return
		}

		release, err := g.mgr.Acquire(r.Context(), resolved.ID, resolved.Entry)
		if err != nil {
			g.writeManagerError(w, err)
			return
		}
		defer release()

		req := &backend.Request{
			Path:   path,
			Body:   body,
			Stream: probe.Stream,
			Header: r.Header,
		}
		be := g.reg.Get(resolved.Entry.Backend)
		if be == nil {
			writeError(w, http.StatusInternalServerError, "backend_missing", "no backend for "+resolved.Entry.Backend)
			return
		}

		result, ferr := be.Forward(r.Context(), resolved.Entry, req, w)
		g.logInference(r, path, token, resolved, probe.Stream, result, ferr, start)
		// Forward has already written the response (status, body, or stream). On
		// error mid-stream there is nothing more we can safely write.
	}
}

// handleStatus reports full residency for the admin plane.
func (g *Gateway) handleStatus(w http.ResponseWriter, r *http.Request) {
	if err := g.authenticateAdmin(r); err != nil {
		writeAuthError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, g.mgr.Status())
}

// adminModelRequest is the body for /admin/load and /admin/unload.
type adminModelRequest struct {
	Model string `json:"model"`
}

// handleAdminLoad forces a model into the main slot (or resident, if coexist)
// without serving a request. It is the warm-up hook used after provisioning.
func (g *Gateway) handleAdminLoad(w http.ResponseWriter, r *http.Request) {
	if err := g.authenticateAdmin(r); err != nil {
		writeAuthError(w, err)
		return
	}
	var body adminModelRequest
	if err := decodeJSON(r, &body); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_json", err.Error())
		return
	}
	resolved, ok := g.router.Resolve(body.Model)
	if !ok {
		writeError(w, http.StatusNotFound, "model_not_found", "unknown model: "+body.Model)
		return
	}
	release, err := g.mgr.Acquire(r.Context(), resolved.ID, resolved.Entry)
	if err != nil {
		g.writeManagerError(w, err)
		return
	}
	release()
	writeJSON(w, http.StatusOK, map[string]any{"loaded": resolved.ID})
}

// handleAdminUnload drops the named model's backend from VRAM. It is best
// effort: an unreachable backend is reported but not retried here.
func (g *Gateway) handleAdminUnload(w http.ResponseWriter, r *http.Request) {
	if err := g.authenticateAdmin(r); err != nil {
		writeAuthError(w, err)
		return
	}
	var body adminModelRequest
	if err := decodeJSON(r, &body); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_json", err.Error())
		return
	}
	resolved, ok := g.router.Resolve(body.Model)
	if !ok {
		writeError(w, http.StatusNotFound, "model_not_found", "unknown model: "+body.Model)
		return
	}
	be := g.reg.Get(resolved.Entry.Backend)
	if be == nil {
		writeError(w, http.StatusInternalServerError, "backend_missing", "no backend for "+resolved.Entry.Backend)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	if err := be.Unload(ctx, resolved.Entry); err != nil {
		writeError(w, http.StatusBadGateway, "unload_failed", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"unloaded": resolved.ID})
}

// authenticate checks the data-plane bearer token and returns its logging prefix.
func (g *Gateway) authenticate(r *http.Request) (string, *auth.AuthError) {
	token, autherr := g.auth.Check(r)
	if autherr != nil {
		return "", autherr
	}
	return auth.Prefix(token), nil
}

// authenticateAdmin requires the admin token when one is configured. When no
// admin token is set the admin plane is closed entirely (tailnet-only boxes can
// still use the data-plane token by setting admin_token equal to it).
func (g *Gateway) authenticateAdmin(r *http.Request) *auth.AuthError {
	if g.admin == nil {
		return &auth.AuthError{Code: "admin_disabled", Message: "admin plane is not enabled"}
	}
	_, autherr := g.admin.Check(r)
	return autherr
}

func (g *Gateway) writeManagerError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, manager.ErrQueueFull):
		writeError(w, http.StatusServiceUnavailable, "queue_full", "model swap queue is full, retry shortly")
	case errors.Is(err, manager.ErrCoexistBudget):
		writeError(w, http.StatusInsufficientStorage, "vram_budget", err.Error())
	case errors.Is(err, backend.ErrOOM):
		writeError(w, http.StatusInsufficientStorage, "out_of_memory", "backend ran out of VRAM loading the model")
	default:
		writeError(w, http.StatusBadGateway, "load_failed", err.Error())
	}
}

func (g *Gateway) logInference(r *http.Request, path, tokenPrefix string, resolved router.Resolved, stream bool, result *backend.Result, ferr error, start time.Time) {
	rec := observe.Request{
		Method:        r.Method,
		Path:          path,
		Model:         resolved.ID,
		Backend:       resolved.Entry.Backend,
		UpstreamModel: resolved.Entry.UpstreamModel,
		Streamed:      stream,
		TokenPrefix:   tokenPrefix,
		TotalMillis:   time.Since(start).Milliseconds(),
	}
	if result != nil {
		rec.Status = result.Status
		rec.TTFTMillis = result.TTFT.Milliseconds()
		rec.PromptTokens = result.PromptTokens
		rec.CompletionTokens = result.CompletionTokens
	}
	if ferr != nil {
		if rec.Status == 0 {
			rec.Status = http.StatusBadGateway
		}
		g.log.Error("forward_failed", map[string]any{
			"model":   resolved.ID,
			"backend": resolved.Entry.Backend,
			"path":    path,
			"error":   ferr.Error(),
		})
	}
	g.log.LogRequest(rec)
}
