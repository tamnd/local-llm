package backend

import (
	"context"
	"errors"
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
}
