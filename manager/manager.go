// Package manager is the VRAM steward. It guarantees that at most one large
// model occupies the GPU's main slot at a time, swapping models in and out as
// requests target them, while letting small models marked coexist stay resident
// alongside the main one. The swap sequence (drain in-flight requests, unload,
// load, promote, release the queue) is the algorithm from doc 08 section 5.4.
//
// Concurrency model: one mutex plus a condition variable. A request calls
// Acquire to make its model resident and register itself as in-flight; the
// returned release marks it done. A swap claims the swapper role, waits for the
// in-flight count to reach zero (the drain), performs the unload/load off the
// lock (those block on the network), then promotes the new model and wakes
// everyone waiting.
package manager

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/tamnd/local-llm/backend"
	"github.com/tamnd/local-llm/config"
	"github.com/tamnd/local-llm/observe"
)

// Errors the gateway maps to HTTP responses.
var (
	// ErrQueueFull is returned when too many requests pile up behind a swap.
	ErrQueueFull = errors.New("manager: swap queue is full")
	// ErrCoexistBudget is returned when a coexist model would push resident
	// VRAM past the budget ceiling.
	ErrCoexistBudget = errors.New("manager: coexist model exceeds VRAM budget")
)

// Manager owns the residency state. It is safe for concurrent use.
type Manager struct {
	reg      backend.Registry
	log      *observe.Logger
	budget   int // VRAM budget ceiling in MiB
	hotSwap  bool
	timeouts timeouts
	queueMax int

	mu          sync.Mutex
	cond        *sync.Cond
	active      string // current main-slot model id ("" if none)
	activeEntry config.ModelEntry
	swapping    bool
	inflight    int // requests forwarding on the main slot
	queued      int // requests waiting for a swap to finish
	coexist     map[string]coexistState
}

type coexistState struct {
	entry  config.ModelEntry
	vramMB int
}

type timeouts struct {
	load, unload, drain time.Duration
}

// New builds a Manager from the config and the backend registry.
func New(cfg *config.Config, reg backend.Registry, log *observe.Logger) *Manager {
	m := &Manager{
		reg:      reg,
		log:      log,
		budget:   cfg.Manager.VRAMBudgetMB,
		hotSwap:  cfg.Manager.HotSwap,
		queueMax: cfg.Manager.QueueMax,
		coexist:  map[string]coexistState{},
		timeouts: timeouts{
			load:   time.Duration(cfg.Manager.LoadTimeoutS) * time.Second,
			unload: time.Duration(cfg.Manager.UnloadTimeoutS) * time.Second,
			drain:  time.Duration(cfg.Manager.DrainTimeoutS) * time.Second,
		},
	}
	m.cond = sync.NewCond(&m.mu)
	return m
}

// Acquire makes the model named by id/entry resident and registers the caller
// as in-flight. It returns a release function the caller must invoke when the
// request finishes (typically via defer). For coexist models it ensures the
// model is loaded and returns a no-op release, because coexist requests run
// alongside the main slot and must not block its swaps.
func (m *Manager) Acquire(ctx context.Context, id string, entry config.ModelEntry) (func(), error) {
	if entry.Coexist {
		if err := m.ensureCoexist(ctx, id, entry); err != nil {
			return nil, err
		}
		return func() {}, nil
	}

	m.mu.Lock()
	for {
		switch {
		case m.swapping:
			if m.queued >= m.queueMax {
				m.mu.Unlock()
				return nil, ErrQueueFull
			}
			m.queued++
			m.cond.Wait()
			m.queued--
		case m.active == id:
			m.inflight++
			m.mu.Unlock()
			return m.releaseOnce(), nil
		default:
			// Claim the swapper role and drain in-flight requests.
			m.swapping = true
			from := m.active
			fromEntry := m.activeEntry
			for m.inflight > 0 {
				m.cond.Wait()
			}
			queuedAtStart := m.queued
			m.mu.Unlock()

			err := m.doSwap(ctx, from, fromEntry, id, entry, queuedAtStart)

			m.mu.Lock()
			m.swapping = false
			if err == nil {
				m.active = id
				m.activeEntry = entry
			} else {
				m.active = ""
				m.activeEntry = config.ModelEntry{}
			}
			m.cond.Broadcast()
			if err != nil {
				m.mu.Unlock()
				return nil, err
			}
			// Loop: re-check state (another swap may have raced) before
			// registering as in-flight.
		}
	}
}

// doSwap unloads the outgoing model (when hot-swap is on) and loads the
// incoming one. It runs without the lock held because the backend calls block
// on the network. It logs swap_start and swap_done events (doc 08 section 11).
func (m *Manager) doSwap(ctx context.Context, from string, fromEntry config.ModelEntry, to string, toEntry config.ModelEntry, queued int) error {
	start := time.Now()
	m.log.Info("swap_start", map[string]any{"from_model": from, "to_model": to, "queued_requests": queued})

	var unloadMS int64
	if m.hotSwap && from != "" {
		uctx, cancel := context.WithTimeout(ctx, m.timeouts.unload)
		err := m.backendFor(fromEntry).Unload(uctx, fromEntry)
		cancel()
		unloadMS = time.Since(start).Milliseconds()
		if err != nil {
			// An unload failure is logged but not fatal: the load below may
			// still succeed if the runtime freed VRAM on its own, and failing
			// the swap here would strand the box with no model.
			m.log.Warn("unload_failed", map[string]any{"model": from, "error": err.Error()})
		}
	}

	loadStart := time.Now()
	lctx, cancel := context.WithTimeout(ctx, m.timeouts.load)
	err := m.backendFor(toEntry).Load(lctx, toEntry)
	cancel()
	if err != nil {
		m.log.Error("load_failed", map[string]any{"model": to, "error": err.Error()})
		return fmt.Errorf("load %s: %w", to, err)
	}

	m.log.Info("swap_done", map[string]any{
		"to_model":  to,
		"unload_ms": unloadMS,
		"load_ms":   time.Since(loadStart).Milliseconds(),
		"total_ms":  time.Since(start).Milliseconds(),
	})
	return nil
}

// ensureCoexist loads a coexist model if it is not already resident, after
// checking that adding it keeps total resident VRAM within budget.
func (m *Manager) ensureCoexist(ctx context.Context, id string, entry config.ModelEntry) error {
	m.mu.Lock()
	if _, ok := m.coexist[id]; ok {
		m.mu.Unlock()
		return nil
	}
	if used := m.residentVRAMLocked() + entry.VRAMMB; used > m.budget {
		m.mu.Unlock()
		return fmt.Errorf("%w: %d MiB needed, %d MiB budget", ErrCoexistBudget, used, m.budget)
	}
	m.mu.Unlock()

	lctx, cancel := context.WithTimeout(ctx, m.timeouts.load)
	defer cancel()
	if err := m.backendFor(entry).Load(lctx, entry); err != nil {
		return fmt.Errorf("load coexist %s: %w", id, err)
	}

	m.mu.Lock()
	m.coexist[id] = coexistState{entry: entry, vramMB: entry.VRAMMB}
	m.mu.Unlock()
	return nil
}

// residentVRAMLocked sums the VRAM of the main slot model and every coexist
// model. The caller holds the lock.
func (m *Manager) residentVRAMLocked() int {
	total := 0
	if m.active != "" {
		total += m.activeEntry.VRAMMB
	}
	for _, c := range m.coexist {
		total += c.vramMB
	}
	return total
}

// releaseOnce returns a release function that decrements the in-flight count
// exactly once and wakes any swapper waiting to drain.
func (m *Manager) releaseOnce() func() {
	var once sync.Once
	return func() {
		once.Do(func() {
			m.mu.Lock()
			m.inflight--
			m.cond.Broadcast()
			m.mu.Unlock()
		})
	}
}

func (m *Manager) backendFor(entry config.ModelEntry) backend.Backend {
	return m.reg.Get(entry.Backend)
}

// Status is a snapshot of residency for the /health and /admin/status endpoints.
type Status struct {
	ActiveModel    string   `json:"active_model"`
	State          string   `json:"state"`
	QueueDepth     int      `json:"queue_depth"`
	Inflight       int      `json:"inflight"`
	Coexist        []string `json:"coexist"`
	ResidentVRAMMB int      `json:"resident_vram_mb"`
}

// Status returns the current residency snapshot.
func (m *Manager) Status() Status {
	m.mu.Lock()
	defer m.mu.Unlock()
	state := "active"
	if m.swapping {
		state = "swapping"
	} else if m.active == "" {
		state = "empty"
	}
	coexist := make([]string, 0, len(m.coexist))
	for id := range m.coexist {
		coexist = append(coexist, id)
	}
	return Status{
		ActiveModel:    m.active,
		State:          state,
		QueueDepth:     m.queued,
		Inflight:       m.inflight,
		Coexist:        coexist,
		ResidentVRAMMB: m.residentVRAMLocked(),
	}
}

// ActiveBackend returns the backend id currently serving the main slot, or "".
func (m *Manager) ActiveBackend() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.activeEntry.Backend
}
