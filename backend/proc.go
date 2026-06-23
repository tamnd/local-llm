package backend

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"sync"
	"time"
)

// process is a running runtime subprocess (llama-server or vLLM). The interface
// exists so tests can substitute a fake and exercise the load/unload
// orchestration without launching real binaries.
type process interface {
	// Stop terminates the process and waits for it to exit, freeing its VRAM.
	Stop(ctx context.Context) error
}

// launcher starts a runtime subprocess. The production launcher shells out via
// os/exec; tests inject a fake.
type launcher interface {
	start(ctx context.Context, bin string, args []string) (process, error)
}

// procTable tracks the single subprocess a process-driven backend currently
// owns. Only one model is resident at a time on this box, so the table holds at
// most one process; loading a new model stops the old one first.
type procTable struct {
	mu       sync.Mutex
	launcher launcher
	cur      process
	curModel string
}

func newProcTable() *procTable {
	return &procTable{launcher: execLauncher{}}
}

// swap stops the current process (if any) and starts a new one for model. It is
// the load half of a process-driven backend; unload is stop with an empty
// target. waitReady is called after start to block until the runtime answers
// health checks, so the caller does not forward a request to a process that has
// not finished loading weights.
func (t *procTable) swap(ctx context.Context, model, bin string, args []string, waitReady func(context.Context) error) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.cur != nil && t.curModel == model {
		return nil // already running this model
	}
	if err := t.stopLocked(ctx); err != nil {
		return err
	}
	if bin == "" {
		return fmt.Errorf("backend: no binary configured for model %q", model)
	}
	p, err := t.launcher.start(ctx, bin, args)
	if err != nil {
		return fmt.Errorf("start %s: %w", bin, err)
	}
	if waitReady != nil {
		if err := waitReady(ctx); err != nil {
			_ = p.Stop(ctx)
			return fmt.Errorf("waiting for %s to become ready: %w", model, err)
		}
	}
	t.cur = p
	t.curModel = model
	return nil
}

// stop terminates the current process, if any.
func (t *procTable) stop(ctx context.Context) error {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.stopLocked(ctx)
}

func (t *procTable) stopLocked(ctx context.Context) error {
	if t.cur == nil {
		return nil
	}
	err := t.cur.Stop(ctx)
	t.cur = nil
	t.curModel = ""
	return err
}

// execProcess wraps an *exec.Cmd started detached from the request context so
// that finishing one request does not kill the long-lived runtime.
type execProcess struct {
	cmd *exec.Cmd
}

func (e *execProcess) Stop(ctx context.Context) error {
	if e.cmd == nil || e.cmd.Process == nil {
		return nil
	}
	if err := e.cmd.Process.Kill(); err != nil {
		return err
	}
	done := make(chan error, 1)
	go func() { done <- e.cmd.Wait() }()
	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(10 * time.Second):
		return fmt.Errorf("backend: process did not exit within 10s")
	}
}

// execLauncher is the production launcher. It starts the binary detached from
// the request context (a background context) so the runtime outlives the call
// that started it; the procTable owns its lifecycle from then on.
type execLauncher struct{}

func (execLauncher) start(_ context.Context, bin string, args []string) (process, error) {
	cmd := exec.Command(bin, args...)
	cmd.Env = os.Environ()
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return &execProcess{cmd: cmd}, nil
}

// waitHealthy polls check until it returns nil or the deadline passes. It is the
// readiness gate after a process-driven backend starts.
func waitHealthy(ctx context.Context, timeout time.Duration, check func(context.Context) error) error {
	deadline := time.Now().Add(timeout)
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()
	for {
		if err := check(ctx); err == nil {
			return nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("not healthy after %s", timeout)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
		}
	}
}
