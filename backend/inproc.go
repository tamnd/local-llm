package backend

import (
	"context"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"sync"

	"github.com/tamnd/local-llm/config"
	"github.com/tamnd/local-llm/llama"
)

// InProc is the zero-proxy backend: it loads GGUF models directly into the
// gateway process with the in-process llama engine and serves generation without
// a subprocess or an HTTP hop. Every other adapter forwards over loopback to a
// runtime that owns the model; this one owns the model itself (spec 2065 doc 16).
//
// It holds a map of resident runners keyed by upstream model name, not a single
// slot, because the manager keeps a coexisting embedding model loaded alongside
// the main model and both can be inproc. A swap of the main model unloads only
// its runner; the coexisting one stays put. Each runner wraps a single-stream
// llama context, so a per-runner mutex serializes that model's generations while
// different models run concurrently.
type InProc struct {
	version   string
	newRunner runnerFactory

	mu      sync.Mutex
	runners map[string]*runnerHandle
}

// runnerHandle is one resident model plus the lock that serializes its decode
// loop. The llama context behind it is single-stream.
type runnerHandle struct {
	runner llama.Runner
	genMu  sync.Mutex
}

// runnerFactory builds a runner from load params. It is a field so tests can
// inject a fake runner without a GPU; production uses llama.New.
type runnerFactory func(llama.Params) (llama.Runner, error)

// newInProc builds the adapter with the real llama engine as its factory.
func newInProc(version string) *InProc {
	return &InProc{
		version:   version,
		newRunner: llama.New,
		runners:   make(map[string]*runnerHandle),
	}
}

// ID returns the backend id "inproc".
func (i *InProc) ID() string { return config.BackendInproc }

// Load makes entry.UpstreamModel resident by constructing a runner from its
// params and storing it under the upstream name. A model already resident is a
// no-op. The load happens off the adapter lock because it is slow (weights move
// to VRAM); a double-checked insert handles the rare concurrent load of the same
// model.
func (i *InProc) Load(ctx context.Context, entry config.ModelEntry) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	i.mu.Lock()
	if _, ok := i.runners[entry.UpstreamModel]; ok {
		i.mu.Unlock()
		return nil
	}
	i.mu.Unlock()

	runner, err := i.newRunner(paramsFor(entry))
	if err != nil {
		if isOOM([]byte(err.Error())) {
			return fmt.Errorf("%w: %s", ErrOOM, err.Error())
		}
		return err
	}

	i.mu.Lock()
	if _, ok := i.runners[entry.UpstreamModel]; ok {
		// Lost the race: another Load got there first. Drop ours.
		i.mu.Unlock()
		_ = runner.Close()
		return nil
	}
	i.runners[entry.UpstreamModel] = &runnerHandle{runner: runner}
	i.mu.Unlock()
	return nil
}

// Unload closes the runner for entry.UpstreamModel and frees its VRAM. The
// manager drains in-flight requests before unloading the main model, so no
// Forward is mid-generation on this runner when Unload runs.
func (i *InProc) Unload(_ context.Context, entry config.ModelEntry) error {
	i.mu.Lock()
	h, ok := i.runners[entry.UpstreamModel]
	delete(i.runners, entry.UpstreamModel)
	i.mu.Unlock()
	if !ok {
		return nil
	}
	return h.runner.Close()
}

// Healthy reports whether the model is resident. For an in-process engine there
// is no socket to probe; residency is health.
func (i *InProc) Healthy(_ context.Context, entry config.ModelEntry) error {
	i.mu.Lock()
	_, ok := i.runners[entry.UpstreamModel]
	i.mu.Unlock()
	if !ok {
		return fmt.Errorf("%w: model %q not resident", ErrUnreachable, entry.UpstreamModel)
	}
	return nil
}

// Forward runs one OpenAI-compatible request against the resident runner and
// writes the response (JSON or SSE) to w. It serializes on the runner's genMu so
// a single context never sees two concurrent decode loops.
func (i *InProc) Forward(ctx context.Context, entry config.ModelEntry, req *Request, w http.ResponseWriter) (*Result, error) {
	i.mu.Lock()
	h, ok := i.runners[entry.UpstreamModel]
	i.mu.Unlock()
	if !ok {
		return nil, fmt.Errorf("%w: model %q not resident", ErrUnreachable, entry.UpstreamModel)
	}

	h.genMu.Lock()
	defer h.genMu.Unlock()

	fingerprint := fmt.Sprintf("llmgw-%s-%s", i.version, config.BackendInproc)
	switch {
	case strings.HasSuffix(req.Path, "/chat/completions"):
		return serveChat(ctx, h.runner, entry, req, w, fingerprint)
	case strings.HasSuffix(req.Path, "/completions"):
		return serveCompletion(ctx, h.runner, entry, req, w, fingerprint)
	case strings.HasSuffix(req.Path, "/embeddings"):
		return serveEmbeddings(ctx, h.runner, entry, req, w, fingerprint)
	default:
		return nil, fmt.Errorf("inproc: unsupported path %q", req.Path)
	}
}

// paramsFor builds the engine load params from a model entry. The tuning levers
// default to the box-tested values from doc 15 (full GPU offload, flash
// attention, q8 KV) and are overridable per model. Embedding models force f16 KV
// and no flash attention, the configuration the pooling path expects.
func paramsFor(entry config.ModelEntry) llama.Params {
	embedding := paramBool(entry.Params, "embedding", false)
	p := llama.Params{
		ModelPath:  paramString(entry.Params, "model_path", ""),
		DraftPath:  paramString(entry.Params, "draft_path", ""),
		NCtx:       paramInt(entry.Params, "n_ctx", 0),
		NGPULayers: paramInt(entry.Params, "n_gpu_layers", 99),
		MainGPU:    paramInt(entry.Params, "main_gpu", 0),
		FlashAttn:  paramBool(entry.Params, "flash_attn", !embedding),
		CacheTypeK: paramString(entry.Params, "cache_type_k", defaultCacheType(embedding)),
		CacheTypeV: paramString(entry.Params, "cache_type_v", defaultCacheType(embedding)),
		Embedding:  embedding,
		DraftMax:   paramInt(entry.Params, "draft_max", 4),
		DraftMin:   paramInt(entry.Params, "draft_min", 0),
		DraftPMin:  paramFloat32(entry.Params, "draft_p_min", 0),
	}
	return p
}

// defaultCacheType picks the KV cache quantization default: f16 for embedding
// models (pooling reads the cache and wants full precision), q8_0 for generative
// models (half the cache for free, matched K and V for the fused flash path).
func defaultCacheType(embedding bool) string {
	if embedding {
		return "f16"
	}
	return "q8_0"
}

// paramFloat32 reads a float param, accepting the several numeric shapes the YAML
// decoder produces.
func paramFloat32(params map[string]any, key string, def float32) float32 {
	if params == nil {
		return def
	}
	switch v := params[key].(type) {
	case float64:
		return float32(v)
	case float32:
		return v
	case int:
		return float32(v)
	case string:
		if f, err := strconv.ParseFloat(v, 32); err == nil {
			return float32(f)
		}
	}
	return def
}
