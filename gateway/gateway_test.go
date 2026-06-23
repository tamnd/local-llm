package gateway

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tamnd/local-llm/backend"
	"github.com/tamnd/local-llm/config"
	"github.com/tamnd/local-llm/manager"
	"github.com/tamnd/local-llm/observe"
	"github.com/tamnd/local-llm/router"
)

// fakeBackend echoes a canned chat completion and records the last request it saw.
type fakeBackend struct {
	id       string
	lastBody []byte
	lastPath string
}

func (f *fakeBackend) ID() string                                       { return f.id }
func (f *fakeBackend) Load(context.Context, config.ModelEntry) error    { return nil }
func (f *fakeBackend) Unload(context.Context, config.ModelEntry) error  { return nil }
func (f *fakeBackend) Healthy(context.Context, config.ModelEntry) error { return nil }
func (f *fakeBackend) Forward(_ context.Context, _ config.ModelEntry, req *backend.Request, w http.ResponseWriter) (*backend.Result, error) {
	f.lastBody = req.Body
	f.lastPath = req.Path
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = io.WriteString(w, `{"id":"cmpl-1","choices":[{"finish_reason":"stop"}]}`)
	return &backend.Result{Status: 200, PromptTokens: 7, CompletionTokens: 3}, nil
}

func testGateway(t *testing.T, fb *fakeBackend) http.Handler {
	t.Helper()
	cfg := &config.Config{
		Auth:         config.Auth{Tokens: []string{"secret-token"}, AdminToken: "admin-token"},
		DefaultModel: "default",
		Models: map[string]config.ModelEntry{
			"qwen3-30b-a3b": {Backend: "fake", BaseURL: "u", UpstreamModel: "qwen3:30b-a3b", VRAMMB: 20000},
		},
		Aliases: map[string]string{"default": "qwen3-30b-a3b"},
		Manager: config.Manager{VRAMBudgetMB: 22528, HotSwap: true, QueueMax: 4, LoadTimeoutS: 5, UnloadTimeoutS: 5, DrainTimeoutS: 5},
	}
	reg := backend.Registry{"fake": fb}
	log := observe.New(io.Discard, "error")
	mgr := manager.New(cfg, reg, log)
	rt := router.New(cfg)
	return New(cfg, rt, mgr, reg, log, "test").Handler()
}

func do(h http.Handler, method, path, token, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func TestHealthNeedsNoToken(t *testing.T) {
	h := testGateway(t, &fakeBackend{id: "fake"})
	rec := do(h, "GET", "/healthz", "", "")
	if rec.Code != http.StatusOK {
		t.Fatalf("health status = %d", rec.Code)
	}
}

func TestChatCompletionFlows(t *testing.T) {
	fb := &fakeBackend{id: "fake"}
	h := testGateway(t, fb)
	rec := do(h, "POST", "/v1/chat/completions", "secret-token", `{"model":"default","stream":false}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if fb.lastPath != "/v1/chat/completions" {
		t.Errorf("backend saw path %q", fb.lastPath)
	}
	if !strings.Contains(rec.Body.String(), "cmpl-1") {
		t.Errorf("body not forwarded: %s", rec.Body.String())
	}
}

func TestUnauthorizedRejected(t *testing.T) {
	h := testGateway(t, &fakeBackend{id: "fake"})
	rec := do(h, "POST", "/v1/chat/completions", "", `{"model":"default"}`)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("missing token status = %d, want 401", rec.Code)
	}
	rec = do(h, "POST", "/v1/chat/completions", "wrong-token", `{"model":"default"}`)
	if rec.Code != http.StatusForbidden {
		t.Fatalf("bad token status = %d, want 403", rec.Code)
	}
}

func TestUnknownModel404(t *testing.T) {
	h := testGateway(t, &fakeBackend{id: "fake"})
	rec := do(h, "POST", "/v1/chat/completions", "secret-token", `{"model":"gpt-4o"}`)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("unknown model status = %d, want 404", rec.Code)
	}
}

func TestModelsListNoAliases(t *testing.T) {
	h := testGateway(t, &fakeBackend{id: "fake"})
	rec := do(h, "GET", "/v1/models", "secret-token", "")
	if rec.Code != http.StatusOK {
		t.Fatalf("models status = %d", rec.Code)
	}
	var out struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(out.Data) != 1 || out.Data[0].ID != "qwen3-30b-a3b" {
		t.Errorf("models = %+v, want single canonical id", out.Data)
	}
}

func TestAdminStatusNeedsAdminToken(t *testing.T) {
	h := testGateway(t, &fakeBackend{id: "fake"})
	// Data-plane token must not open the admin plane.
	if rec := do(h, "GET", "/admin/status", "secret-token", ""); rec.Code != http.StatusForbidden {
		t.Fatalf("data token on admin = %d, want 403", rec.Code)
	}
	if rec := do(h, "GET", "/admin/status", "admin-token", ""); rec.Code != http.StatusOK {
		t.Fatalf("admin token on admin = %d, want 200", rec.Code)
	}
}

func TestAdminLoadWarmsModel(t *testing.T) {
	h := testGateway(t, &fakeBackend{id: "fake"})
	rec := do(h, "POST", "/admin/load", "admin-token", `{"model":"default"}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("admin load status = %d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "qwen3-30b-a3b") {
		t.Errorf("load response = %s", rec.Body.String())
	}
}
