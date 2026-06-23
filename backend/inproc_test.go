package backend

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tamnd/local-llm/config"
	"github.com/tamnd/local-llm/llama"
)

// fakeRunner stands in for the cgo engine so the adapter can be tested without a
// GPU. It replays a fixed list of pieces and reports canned stats.
type fakeRunner struct {
	pieces []string
	stats  llama.Stats
	closed bool
	embed  []float32
}

func (f *fakeRunner) run(emit func(string) bool) (llama.Stats, error) {
	for _, p := range f.pieces {
		if !emit(p) {
			break
		}
	}
	return f.stats, nil
}

func (f *fakeRunner) Chat(_ context.Context, _ []llama.Message, _ llama.Sampling, emit func(string) bool) (llama.Stats, error) {
	return f.run(emit)
}

func (f *fakeRunner) Complete(_ context.Context, _ string, _ llama.Sampling, emit func(string) bool) (llama.Stats, error) {
	return f.run(emit)
}

func (f *fakeRunner) Embed(_ context.Context, _ string) ([]float32, error) {
	return f.embed, nil
}

func (f *fakeRunner) Close() error {
	f.closed = true
	return nil
}

// newTestInProc builds an InProc adapter whose factory returns the given runner,
// so no GGUF or CUDA is involved.
func newTestInProc(r llama.Runner) *InProc {
	return &InProc{
		version:   "test",
		newRunner: func(llama.Params) (llama.Runner, error) { return r, nil },
		runners:   make(map[string]*runnerHandle),
	}
}

func inprocEntry() config.ModelEntry {
	return config.ModelEntry{
		Backend:       config.BackendInproc,
		UpstreamModel: "qwen3.6-27b",
		Params:        map[string]any{"model_path": "/models/qwen3.6-27b.gguf"},
	}
}

func TestInProcLoadUnloadResidency(t *testing.T) {
	fr := &fakeRunner{}
	i := newTestInProc(fr)
	entry := inprocEntry()

	if err := i.Healthy(context.Background(), entry); err == nil {
		t.Fatal("model should not be resident before load")
	}
	if err := i.Load(context.Background(), entry); err != nil {
		t.Fatalf("load: %v", err)
	}
	if err := i.Healthy(context.Background(), entry); err != nil {
		t.Fatalf("model should be resident after load: %v", err)
	}
	// A second load is a no-op and must not build a second runner.
	if err := i.Load(context.Background(), entry); err != nil {
		t.Fatalf("second load: %v", err)
	}
	if err := i.Unload(context.Background(), entry); err != nil {
		t.Fatalf("unload: %v", err)
	}
	if !fr.closed {
		t.Error("unload did not close the runner")
	}
	if err := i.Healthy(context.Background(), entry); err == nil {
		t.Error("model should not be resident after unload")
	}
}

func TestInProcChatBlocking(t *testing.T) {
	fr := &fakeRunner{
		pieces: []string{"Hello", ", ", "world"},
		stats:  llama.Stats{PromptTokens: 11, CompletionTokens: 3, StopReason: "stop"},
	}
	i := newTestInProc(fr)
	entry := inprocEntry()
	if err := i.Load(context.Background(), entry); err != nil {
		t.Fatalf("load: %v", err)
	}

	req := &Request{Path: "/v1/chat/completions", Body: []byte(`{"model":"qwen3.6-27b","messages":[{"role":"user","content":"hi"}]}`)}
	rec := httptest.NewRecorder()
	res, err := i.Forward(context.Background(), entry, req, rec)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	if res.Status != http.StatusOK || res.CompletionTokens != 3 {
		t.Errorf("result = %+v", res)
	}

	var obj struct {
		Object            string `json:"object"`
		SystemFingerprint string `json:"system_fingerprint"`
		Choices           []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			TotalTokens int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &obj); err != nil {
		t.Fatalf("decode: %v body=%s", err, rec.Body.String())
	}
	if obj.Object != "chat.completion" {
		t.Errorf("object = %q", obj.Object)
	}
	if obj.SystemFingerprint != "llmgw-test-inproc" {
		t.Errorf("fingerprint = %q", obj.SystemFingerprint)
	}
	if len(obj.Choices) != 1 || obj.Choices[0].Message.Content != "Hello, world" {
		t.Errorf("content = %+v", obj.Choices)
	}
	if obj.Choices[0].FinishReason != "stop" {
		t.Errorf("finish_reason = %q", obj.Choices[0].FinishReason)
	}
	if obj.Usage.TotalTokens != 14 {
		t.Errorf("total_tokens = %d, want 14", obj.Usage.TotalTokens)
	}
}

func TestInProcChatStreaming(t *testing.T) {
	fr := &fakeRunner{
		pieces: []string{"a", "b", "c"},
		stats:  llama.Stats{PromptTokens: 5, CompletionTokens: 3, StopReason: "length"},
	}
	i := newTestInProc(fr)
	entry := inprocEntry()
	if err := i.Load(context.Background(), entry); err != nil {
		t.Fatalf("load: %v", err)
	}

	req := &Request{Path: "/v1/chat/completions", Stream: true, Body: []byte(`{"model":"qwen3.6-27b","stream":true,"messages":[{"role":"user","content":"hi"}]}`)}
	rec := httptest.NewRecorder()
	if _, err := i.Forward(context.Background(), entry, req, rec); err != nil {
		t.Fatalf("forward: %v", err)
	}
	body := rec.Body.String()
	if !strings.Contains(body, "chat.completion.chunk") {
		t.Errorf("missing chunk object: %s", body)
	}
	if !strings.HasSuffix(strings.TrimSpace(body), "data: [DONE]") {
		t.Errorf("stream did not end with [DONE]: %q", body)
	}
	// The three pieces plus the final chunk must each be an SSE data line.
	if n := strings.Count(body, "data: "); n < 4 {
		t.Errorf("expected at least 4 data lines, got %d: %s", n, body)
	}
	if !strings.Contains(body, `"finish_reason":"length"`) {
		t.Errorf("missing finish_reason on final chunk: %s", body)
	}
	if !strings.Contains(body, `"role":"assistant"`) {
		t.Errorf("first delta missing assistant role: %s", body)
	}
}

func TestInProcEmbeddings(t *testing.T) {
	fr := &fakeRunner{embed: []float32{0.1, 0.2, 0.3}}
	i := newTestInProc(fr)
	entry := inprocEntry()
	entry.UpstreamModel = "nomic-embed"
	entry.Params = map[string]any{"model_path": "/models/nomic.gguf", "embedding": true}
	if err := i.Load(context.Background(), entry); err != nil {
		t.Fatalf("load: %v", err)
	}

	req := &Request{Path: "/v1/embeddings", Body: []byte(`{"model":"nomic-embed","input":["one","two"]}`)}
	rec := httptest.NewRecorder()
	if _, err := i.Forward(context.Background(), entry, req, rec); err != nil {
		t.Fatalf("forward: %v", err)
	}
	var obj struct {
		Object string `json:"object"`
		Data   []struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &obj); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if obj.Object != "list" || len(obj.Data) != 2 {
		t.Fatalf("embeddings shape = %+v", obj)
	}
	if len(obj.Data[1].Embedding) != 3 || obj.Data[1].Embedding[0] != 0.1 {
		t.Errorf("vector = %+v", obj.Data[1].Embedding)
	}
}

func TestInProcForwardUnloadedFails(t *testing.T) {
	i := newTestInProc(&fakeRunner{})
	req := &Request{Path: "/v1/chat/completions", Body: []byte(`{}`)}
	if _, err := i.Forward(context.Background(), inprocEntry(), req, httptest.NewRecorder()); err == nil {
		t.Fatal("forward to an unloaded model should fail")
	}
}

func TestParamsForDefaults(t *testing.T) {
	entry := config.ModelEntry{
		UpstreamModel: "m",
		Params:        map[string]any{"model_path": "/m.gguf"},
	}
	p := paramsFor(entry)
	if p.NGPULayers != 99 || !p.FlashAttn || p.CacheTypeK != "q8_0" || p.DraftMax != 4 {
		t.Errorf("generative defaults wrong: %+v", p)
	}

	embed := config.ModelEntry{
		UpstreamModel: "e",
		Params:        map[string]any{"model_path": "/e.gguf", "embedding": true},
	}
	pe := paramsFor(embed)
	if !pe.Embedding || pe.FlashAttn || pe.CacheTypeK != "f16" {
		t.Errorf("embedding defaults wrong: %+v", pe)
	}
}
