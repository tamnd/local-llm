// Package backend holds the per-runtime adapters that the manager and gateway
// talk to. Each runtime (Ollama, llama-server, TabbyAPI, vLLM) speaks an
// OpenAI-compatible dialect on its own loopback port but loads and unloads
// models by a different mechanism; the adapters hide those differences behind
// one interface (doc 08 section 12).
//
// The interface is deliberately small. Forward and Healthy are shared across
// every adapter through the embedded proxy; only Load and Unload carry
// runtime-specific knowledge, which is the whole reason the package exists.
package backend

import (
	"context"
	"errors"
	"net/http"
	"time"

	"github.com/tamnd/local-llm/config"
)

// Request is one OpenAI-compatible call ready to forward. The gateway has
// already rewritten the "model" field in Body to the backend's upstream name,
// so the adapter only has to proxy it.
type Request struct {
	Path   string      // e.g. "/v1/chat/completions"
	Body   []byte      // request body, model field rewritten to upstream
	Stream bool        // whether the client asked for SSE
	Header http.Header // client headers to forward (Authorization is stripped)
}

// Backend is the contract between the manager and a specific runtime. A single
// adapter instance serves every model routed to its runtime; per-model details
// arrive through the config.ModelEntry argument.
type Backend interface {
	// ID returns the backend id ("ollama", "llama", "tabby", "vllm").
	ID() string

	// Load makes entry.UpstreamModel resident in VRAM. For API-driven runtimes
	// (Ollama, TabbyAPI) this is a call; for process-driven runtimes
	// (llama-server, vLLM) it starts the process.
	Load(ctx context.Context, entry config.ModelEntry) error

	// Unload releases whatever model this backend currently holds.
	Unload(ctx context.Context, entry config.ModelEntry) error

	// Forward proxies req to the backend and writes the normalized
	// OpenAI-format response to w. For streaming requests it copies SSE chunks
	// as they arrive without buffering the whole response. The Result carries
	// the metrics the gateway logs (TTFT, token counts, upstream status).
	Forward(ctx context.Context, entry config.ModelEntry, req *Request, w http.ResponseWriter) (*Result, error)

	// Healthy returns nil if the backend at entry.BaseURL is reachable.
	Healthy(ctx context.Context, entry config.ModelEntry) error
}

// Result is what Forward measured while proxying a request. Fields are
// best-effort: a backend that omits a usage object leaves the token counts at
// zero, and the gateway logs them as such.
type Result struct {
	Status           int           // HTTP status written to the client
	TTFT             time.Duration // time from request send to first response byte
	PromptTokens     int
	CompletionTokens int
}

// Sentinel errors the manager and gateway switch on. ErrOOM maps to a 503 with
// the load_failed_oom code; ErrUnreachable maps to a 502.
var (
	ErrOOM         = errors.New("backend: out of memory loading model")
	ErrUnreachable = errors.New("backend: runtime unreachable")
)

// Registry maps a backend id to its adapter. The gateway builds one Registry at
// startup from the config and hands it to the manager.
type Registry map[string]Backend

// NewRegistry constructs the standard set of adapters sharing one HTTP client
// for forwarding. version is stamped into the system_fingerprint the proxy
// injects (doc 08 section 7.2).
func NewRegistry(version string) Registry {
	p := newProxy(version)
	return Registry{
		config.BackendOllama: &Ollama{proxy: p},
		config.BackendLlama:  &Llama{proxy: p, procs: newProcTable()},
		config.BackendTabby:  &Tabby{proxy: p},
		config.BackendVLLM:   &VLLM{proxy: p, procs: newProcTable()},
		config.BackendInproc: newInProc(version),
	}
}

// Get returns the adapter for id, or nil if the id is unknown. Config
// validation guarantees every model's backend id is known, so a nil here is a
// programming error, not a config error.
func (r Registry) Get(id string) Backend {
	return r[id]
}
