package backend

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tamnd/local-llm/config"
)

// fakeProcess records whether it was stopped.
type fakeProcess struct{ stopped bool }

func (f *fakeProcess) Stop(context.Context) error { f.stopped = true; return nil }

// fakeLauncher hands out fakeProcess instances and counts starts.
type fakeLauncher struct {
	started  []string
	last     *fakeProcess
	failNext bool
}

func (l *fakeLauncher) start(_ context.Context, bin string, _ []string) (process, error) {
	if l.failNext {
		return nil, errors.New("boom")
	}
	l.started = append(l.started, bin)
	l.last = &fakeProcess{}
	return l.last, nil
}

func TestProcTableSwapStopsPrevious(t *testing.T) {
	fl := &fakeLauncher{}
	tbl := &procTable{launcher: fl}

	if err := tbl.swap(context.Background(), "modelA", "llama-server", nil, nil); err != nil {
		t.Fatalf("swap A: %v", err)
	}
	first := fl.last

	// Same model again is a no-op: no new process.
	if err := tbl.swap(context.Background(), "modelA", "llama-server", nil, nil); err != nil {
		t.Fatalf("swap A again: %v", err)
	}
	if len(fl.started) != 1 {
		t.Errorf("same-model swap should not restart: starts=%d", len(fl.started))
	}

	// Different model stops the first and starts a second.
	if err := tbl.swap(context.Background(), "modelB", "llama-server", nil, nil); err != nil {
		t.Fatalf("swap B: %v", err)
	}
	if !first.stopped {
		t.Error("previous process should have been stopped on swap")
	}
	if len(fl.started) != 2 {
		t.Errorf("starts=%d, want 2", len(fl.started))
	}

	if err := tbl.stop(context.Background()); err != nil {
		t.Fatalf("stop: %v", err)
	}
	if !fl.last.stopped {
		t.Error("stop should terminate the current process")
	}
}

func TestProcTableSwapNoBinary(t *testing.T) {
	tbl := &procTable{launcher: &fakeLauncher{}}
	err := tbl.swap(context.Background(), "m", "", nil, nil)
	if err == nil {
		t.Fatal("want error when no binary is configured")
	}
}

func TestLlamaArgsFromParams(t *testing.T) {
	entry := config.ModelEntry{
		BaseURL: "http://127.0.0.1:8080", UpstreamModel: "/models/r1.gguf",
		Params: map[string]any{"n_ctx": 16384, "flash_attn": true},
	}
	joined := strings.Join(buildLlamaArgs(entry), " ")
	for _, want := range []string{"--model /models/r1.gguf", "--port 8080", "--ctx-size 16384", "--flash-attn", "--cache-type-k q8_0"} {
		if !strings.Contains(joined, want) {
			t.Errorf("args missing %q; got %s", want, joined)
		}
	}
	// No draft configured means no speculative flags at all.
	if strings.Contains(joined, "--model-draft") {
		t.Errorf("draft flags emitted without params.draft_path; got %s", joined)
	}
	// Likewise for the optional multimodal and sampling knobs: a text-only entry
	// must produce the same command line it did before those were wired up.
	for _, dead := range []string{"--mmproj", "--temp", "--top-p", "--top-k"} {
		if strings.Contains(joined, dead) {
			t.Errorf("emitted %s for an entry that does not set it; got %s", dead, joined)
		}
	}
}

// TestLlamaArgsVisionAndSampling covers the Unsloth Muse Glimmer recipe: the
// vision projector plus the model card's sampling defaults. The sampling values
// have to survive YAML decoding whether or not the operator quoted them, which
// is why 1.0 arrives here as a float64 and 64 as an int.
func TestLlamaArgsVisionAndSampling(t *testing.T) {
	entry := config.ModelEntry{
		BaseURL: "http://127.0.0.1:8080", UpstreamModel: "/models/muse.gguf",
		Params: map[string]any{
			"mmproj_path": "/models/mmproj-BF16.gguf",
			"temp":        1.0,
			"top_p":       "0.95",
			"top_k":       64,
		},
	}
	args := buildLlamaArgs(entry)
	if !hasFlagValue(args, "--mmproj", "/models/mmproj-BF16.gguf") {
		t.Errorf("missing --mmproj; got %v", args)
	}
	for _, want := range [][2]string{{"--temp", "1"}, {"--top-p", "0.95"}, {"--top-k", "64"}} {
		if !hasFlagValue(args, want[0], want[1]) {
			t.Errorf("missing %s %s; got %v", want[0], want[1], args)
		}
	}
}

// TestLlamaArgsDraft covers the speculative-decoding path: a configured
// draft_path emits --model-draft plus its tuning flags, with draft_max and
// draft_min overridable per model (Muse Glimmer's DFlash head).
func TestLlamaArgsDraft(t *testing.T) {
	entry := config.ModelEntry{
		BaseURL: "http://127.0.0.1:8080", UpstreamModel: "/models/muse.gguf",
		Params: map[string]any{
			"draft_path": "/models/dflash.gguf",
			"draft_max":  8,
		},
	}
	args := buildLlamaArgs(entry)
	if !hasFlagValue(args, "--spec-draft-model", "/models/dflash.gguf") {
		t.Errorf("missing --spec-draft-model; got %v", args)
	}
	if !hasFlagValue(args, "--spec-draft-n-max", "8") {
		t.Errorf("draft_max override not applied; got %v", args)
	}
	// Unset draft knobs fall back to the documented defaults.
	if !hasFlagValue(args, "--spec-draft-n-min", "1") || !hasFlagValue(args, "--spec-draft-ngl", "99") {
		t.Errorf("draft defaults missing; got %v", args)
	}
	// The removed spellings must never be emitted: llama-server exits on them.
	for _, dead := range []string{"--draft-max", "--draft-min"} {
		if hasFlag(args, dead) {
			t.Errorf("emitted removed flag %s; got %v", dead, args)
		}
	}
}

// TestLlamaLoadAdoptsHealthyServer covers the WSL2 path: when llama-server is
// already answering /health at base_url, Load adopts it rather than trying to
// exec a Linux binary from the Windows-host gateway.
func TestLlamaLoadAdoptsHealthyServer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	fl := &fakeLauncher{}
	l := &Llama{proxy: newProxy("test"), procs: &procTable{launcher: fl}}
	entry := config.ModelEntry{BaseURL: srv.URL, UpstreamModel: "/models/muse.gguf"}
	if err := l.Load(context.Background(), entry); err != nil {
		t.Fatalf("Load of a healthy server should succeed by adoption, got %v", err)
	}
	if len(fl.started) != 0 {
		t.Errorf("expected no spawn when the server is already healthy, launched %v", fl.started)
	}
}
