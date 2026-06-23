package backend

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tamnd/local-llm/config"
)

func TestForwardNonStreamingNormalizes(t *testing.T) {
	// A backend that returns a TabbyAPI-style "eos" finish reason and no
	// system_fingerprint. The proxy should canonicalize both.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("unexpected path %q", r.URL.Path)
		}
		json.NewEncoder(w).Encode(map[string]any{
			"id":      "chatcmpl-1",
			"object":  "chat.completion",
			"choices": []any{map[string]any{"index": 0, "finish_reason": "eos", "message": map[string]any{"role": "assistant", "content": "hi"}}},
			"usage":   map[string]any{"prompt_tokens": 5.0, "completion_tokens": 3.0, "total_tokens": 8.0},
		})
	}))
	defer srv.Close()

	p := newProxy("0.1.0")
	rec := httptest.NewRecorder()
	res, err := p.forward(context.Background(), "tabby", srv.URL,
		&Request{Path: "/v1/chat/completions", Body: []byte(`{"model":"x"}`)}, rec)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	if res.Status != 200 || res.PromptTokens != 5 || res.CompletionTokens != 3 {
		t.Errorf("result = %+v, want 200/5/3", res)
	}

	var got map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("response not JSON: %v", err)
	}
	choice := got["choices"].([]any)[0].(map[string]any)
	if choice["finish_reason"] != "stop" {
		t.Errorf("finish_reason = %v, want normalized to stop", choice["finish_reason"])
	}
	if got["system_fingerprint"] != "llmgw-0.1.0-tabby" {
		t.Errorf("system_fingerprint = %v, want injected", got["system_fingerprint"])
	}
}

func TestForwardStreamingCopiesAndNormalizes(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fl := w.(http.Flusher)
		w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"},\"finish_reason\":null}]}\n\n"))
		fl.Flush()
		w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"max_length\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":2}}\n\n"))
		fl.Flush()
		w.Write([]byte("data: [DONE]\n\n"))
		fl.Flush()
	}))
	defer srv.Close()

	p := newProxy("0.1.0")
	rec := httptest.NewRecorder()
	res, err := p.forward(context.Background(), "ollama", srv.URL,
		&Request{Path: "/v1/chat/completions", Body: []byte(`{}`), Stream: true}, rec)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	out := rec.Body.String()
	if !strings.Contains(out, "data: [DONE]") {
		t.Errorf("stream missing DONE sentinel: %q", out)
	}
	if !strings.Contains(out, `"finish_reason":"length"`) {
		t.Errorf("max_length not normalized to length: %q", out)
	}
	if res.CompletionTokens != 2 {
		t.Errorf("completion tokens = %d, want 2 from final chunk", res.CompletionTokens)
	}
}

func TestForwardUnreachable(t *testing.T) {
	p := newProxy("0.1.0")
	rec := httptest.NewRecorder()
	_, err := p.forward(context.Background(), "ollama", "http://127.0.0.1:1",
		&Request{Path: "/v1/chat/completions", Body: []byte(`{}`)}, rec)
	if err == nil {
		t.Fatal("want error for unreachable backend")
	}
}

func TestOllamaLoadUnload(t *testing.T) {
	var loads, unloads int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		switch body["keep_alive"].(type) {
		case float64:
			unloads++ // keep_alive: 0
		default:
			loads++ // keep_alive: "30m"
		}
		w.WriteHeader(200)
	}))
	defer srv.Close()

	o := &Ollama{proxy: newProxy("t")}
	entry := config.ModelEntry{Backend: "ollama", BaseURL: srv.URL, UpstreamModel: "qwen3:30b-a3b"}
	if err := o.Load(context.Background(), entry); err != nil {
		t.Fatalf("Load: %v", err)
	}
	if err := o.Unload(context.Background(), entry); err != nil {
		t.Fatalf("Unload: %v", err)
	}
	if loads != 1 || unloads != 1 {
		t.Errorf("loads=%d unloads=%d, want 1/1", loads, unloads)
	}
}

// TestOllamaLoadEmbedderFallsBack covers an embedding-only model: /api/generate
// is rejected, so Load must warm it through /api/embed instead.
func TestOllamaLoadEmbedderFallsBack(t *testing.T) {
	var sawGenerate, sawEmbed bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/generate":
			sawGenerate = true
			http.Error(w, `{"error":"\"nomic-embed-text\" does not support generate"}`, http.StatusBadRequest)
		case "/api/embed":
			sawEmbed = true
			w.WriteHeader(200)
			_, _ = w.Write([]byte(`{"embeddings":[[0.1]]}`))
		default:
			w.WriteHeader(404)
		}
	}))
	defer srv.Close()

	o := &Ollama{proxy: newProxy("t")}
	entry := config.ModelEntry{Backend: "ollama", BaseURL: srv.URL, UpstreamModel: "nomic-embed-text", Coexist: true}
	if err := o.Load(context.Background(), entry); err != nil {
		t.Fatalf("Load fell through embed fallback: %v", err)
	}
	if !sawGenerate || !sawEmbed {
		t.Errorf("expected generate then embed: generate=%v embed=%v", sawGenerate, sawEmbed)
	}
}

// TestOllamaLoadEmbedFallbackPropagatesOOM makes sure a real OOM on the generate
// warm-up is surfaced, not masked by an embed retry.
func TestOllamaLoadEmbedFallbackPropagatesOOM(t *testing.T) {
	var sawEmbed bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/embed" {
			sawEmbed = true
		}
		http.Error(w, `{"error":"CUDA error: out of memory"}`, http.StatusInternalServerError)
	}))
	defer srv.Close()

	o := &Ollama{proxy: newProxy("t")}
	entry := config.ModelEntry{Backend: "ollama", BaseURL: srv.URL, UpstreamModel: "qwen2.5-coder:32b"}
	err := o.Load(context.Background(), entry)
	if !errors.Is(err, ErrOOM) {
		t.Fatalf("Load err = %v, want ErrOOM", err)
	}
	if sawEmbed {
		t.Error("OOM should short-circuit before the embed fallback")
	}
}

func TestTabbyLoadUnloadCarriesAdminKey(t *testing.T) {
	var sawAuth string
	var loadPath, unloadPath bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sawAuth = r.Header.Get("Authorization")
		switch r.URL.Path {
		case "/v1/model/load":
			loadPath = true
		case "/v1/model/unload":
			unloadPath = true
		}
		w.WriteHeader(200)
	}))
	defer srv.Close()

	tb := &Tabby{proxy: newProxy("t")}
	entry := config.ModelEntry{
		Backend: "tabby", BaseURL: srv.URL, UpstreamModel: "Qwen3.6-27B-EXL3",
		Params: map[string]any{"admin_key": "sk-admin-xyz", "max_seq_len": 16384},
	}
	if err := tb.Load(context.Background(), entry); err != nil {
		t.Fatalf("Load: %v", err)
	}
	if err := tb.Unload(context.Background(), entry); err != nil {
		t.Fatalf("Unload: %v", err)
	}
	if !loadPath || !unloadPath {
		t.Errorf("expected both load and unload calls: load=%v unload=%v", loadPath, unloadPath)
	}
	if sawAuth != "Bearer sk-admin-xyz" {
		t.Errorf("admin key not forwarded: %q", sawAuth)
	}
}

func TestPostJSONMapsOOM(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(500)
		w.Write([]byte(`{"error":"CUDA out of memory: tried to allocate 2GB"}`))
	}))
	defer srv.Close()
	p := newProxy("t")
	err := p.postJSON(context.Background(), srv.URL, "/v1/model/load", []byte("{}"), "")
	if err == nil || !strings.Contains(err.Error(), "out of memory") {
		t.Errorf("want ErrOOM, got %v", err)
	}
}
