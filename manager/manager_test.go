package manager

import (
	"context"
	"errors"
	"io"
	"net/http"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/tamnd/local-llm/backend"
	"github.com/tamnd/local-llm/config"
	"github.com/tamnd/local-llm/observe"
)

// fakeBackend records load/unload calls. loadHook, if set, runs inside Load so a
// test can block to observe intermediate states.
type fakeBackend struct {
	id       string
	loads    int32
	unloads  int32
	loadHook func()
	loadErr  error
}

func (f *fakeBackend) ID() string { return f.id }
func (f *fakeBackend) Load(ctx context.Context, _ config.ModelEntry) error {
	if f.loadHook != nil {
		f.loadHook()
	}
	if f.loadErr != nil {
		return f.loadErr
	}
	atomic.AddInt32(&f.loads, 1)
	return nil
}
func (f *fakeBackend) Unload(context.Context, config.ModelEntry) error {
	atomic.AddInt32(&f.unloads, 1)
	return nil
}
func (f *fakeBackend) Forward(context.Context, config.ModelEntry, *backend.Request, http.ResponseWriter) (*backend.Result, error) {
	return &backend.Result{Status: 200}, nil
}
func (f *fakeBackend) Healthy(context.Context, config.ModelEntry) error { return nil }

func testManager(t *testing.T, fb *fakeBackend) *Manager {
	t.Helper()
	cfg := &config.Config{Manager: config.Manager{
		VRAMBudgetMB: 22528, HotSwap: true, QueueMax: 2,
		LoadTimeoutS: 5, UnloadTimeoutS: 5, DrainTimeoutS: 5,
	}}
	reg := backend.Registry{"fake": fb}
	return New(cfg, reg, observe.New(io.Discard, "error"))
}

func entry(id string, vram int, coexist bool) config.ModelEntry {
	return config.ModelEntry{Backend: "fake", BaseURL: "u", UpstreamModel: id, VRAMMB: vram, Coexist: coexist}
}

func TestAcquireSwapsBetweenModels(t *testing.T) {
	fb := &fakeBackend{id: "fake"}
	m := testManager(t, fb)
	ctx := context.Background()

	rel, err := m.Acquire(ctx, "modelA", entry("a", 20000, false))
	if err != nil {
		t.Fatalf("acquire A: %v", err)
	}
	rel()
	if m.Status().ActiveModel != "modelA" {
		t.Errorf("active = %q, want modelA", m.Status().ActiveModel)
	}

	rel, err = m.Acquire(ctx, "modelB", entry("b", 20000, false))
	if err != nil {
		t.Fatalf("acquire B: %v", err)
	}
	rel()

	if got := atomic.LoadInt32(&fb.loads); got != 2 {
		t.Errorf("loads = %d, want 2 (A then B)", got)
	}
	if got := atomic.LoadInt32(&fb.unloads); got != 1 {
		t.Errorf("unloads = %d, want 1 (A before B)", got)
	}
}

func TestAcquireSameModelNoSwap(t *testing.T) {
	fb := &fakeBackend{id: "fake"}
	m := testManager(t, fb)
	ctx := context.Background()
	for range 3 {
		rel, err := m.Acquire(ctx, "modelA", entry("a", 20000, false))
		if err != nil {
			t.Fatalf("acquire: %v", err)
		}
		rel()
	}
	if got := atomic.LoadInt32(&fb.loads); got != 1 {
		t.Errorf("loads = %d, want 1 (loaded once, reused)", got)
	}
}

func TestSwapDrainsInflight(t *testing.T) {
	fb := &fakeBackend{id: "fake"}
	m := testManager(t, fb)
	ctx := context.Background()

	// Hold an in-flight request on modelA.
	rel, err := m.Acquire(ctx, "modelA", entry("a", 20000, false))
	if err != nil {
		t.Fatalf("acquire A: %v", err)
	}

	swapped := make(chan struct{})
	go func() {
		r, err := m.Acquire(ctx, "modelB", entry("b", 20000, false))
		if err == nil {
			r()
		}
		close(swapped)
	}()

	// The swap must not complete while A is in-flight.
	select {
	case <-swapped:
		t.Fatal("swap completed before in-flight request drained")
	case <-time.After(100 * time.Millisecond):
	}

	rel() // drain A
	select {
	case <-swapped:
	case <-time.After(2 * time.Second):
		t.Fatal("swap did not complete after drain")
	}
}

func TestCoexistBudget(t *testing.T) {
	fb := &fakeBackend{id: "fake"}
	m := testManager(t, fb)
	ctx := context.Background()

	// Main slot uses 20 GB; a 3 GB coexist model would exceed the 22 GB budget.
	rel, _ := m.Acquire(ctx, "main", entry("main", 20000, false))
	rel()

	_, err := m.Acquire(ctx, "embed", entry("embed", 3000, true))
	if err == nil {
		t.Fatal("coexist over budget should be refused")
	}

	// A 300 MB coexist model fits.
	if _, err := m.Acquire(ctx, "embed-small", entry("embed-small", 300, true)); err != nil {
		t.Errorf("small coexist should fit: %v", err)
	}
}

func TestQueueFull(t *testing.T) {
	// loadHook blocks the first swap so others queue behind it.
	release := make(chan struct{})
	var hookOnce sync.Once
	fb := &fakeBackend{id: "fake"}
	fb.loadHook = func() {
		hookOnce.Do(func() { <-release })
	}
	m := testManager(t, fb)
	ctx := context.Background()

	// Kick off the first acquire that will block inside Load (holding swapping).
	go func() {
		r, err := m.Acquire(ctx, "m0", entry("m0", 10000, false))
		if err == nil {
			r()
		}
	}()
	time.Sleep(50 * time.Millisecond) // let m0 enter the blocked Load

	// queueMax is 2: two waiters queue, the third is rejected.
	var rejected int32
	var wg sync.WaitGroup
	for range 4 {
		wg.Go(func() {
			r, err := m.Acquire(ctx, "mX", entry("mX", 10000, false))
			if errors.Is(err, ErrQueueFull) {
				atomic.AddInt32(&rejected, 1)
				return
			}
			if err == nil {
				r()
			}
		})
	}
	time.Sleep(50 * time.Millisecond)
	close(release) // unblock the first swap
	wg.Wait()

	if atomic.LoadInt32(&rejected) == 0 {
		t.Error("expected at least one ErrQueueFull when queue overflowed")
	}
}
