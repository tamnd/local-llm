// Package llama is the in-process inference engine: a thin cgo binding to
// libllama (llama.cpp) that loads a GGUF directly into the gateway process and
// runs the decode loop in-process, with no subprocess and no HTTP hop. It is the
// zero-proxy path argued in spec 2065 doc 16.
//
// The public API in this file is pure Go and is always compiled. The cgo binding
// that implements it lives in cgo.go behind the "llama" build tag, so the engine
// only needs a CUDA-linked libllama on the box where it actually runs. Every
// other build (the macOS dev build, CI) compiles stub.go instead, where New
// returns ErrUnsupported and Available reports false. That keeps the default
// build green without a C toolchain while the real engine is one build tag away.
//
// Nothing outside this package touches a cgo type. Callers see Params, Sampling,
// Message, Stats, and the Runner interface, and the backend adapter in package
// backend drives generation entirely through them.
package llama

import (
	"context"
	"errors"
	"time"
)

// ErrUnsupported is returned by New when the binary was built without the
// in-process engine (the default build, or any build with CGO disabled). Rebuild
// with `-tags llama` and a libllama linked for CUDA to enable it.
var ErrUnsupported = errors.New("llama: in-process engine not built (rebuild with -tags llama)")

// ErrContextFull is returned when a generation would exceed the context window
// the model was loaded with. The caller should surface it as a request error,
// not a backend failure.
var ErrContextFull = errors.New("llama: context window is full")

// Params configure a model load. ModelPath is the only required field; the rest
// carry the tuning levers from doc 15 (full GPU offload, flash attention, q8 KV
// cache) and the optional speculative-decoding draft model.
type Params struct {
	ModelPath  string // path to the GGUF on disk
	DraftPath  string // optional speculative draft GGUF; dense targets only
	NCtx       int    // context window in tokens (0 uses the engine default)
	NGPULayers int    // layers offloaded to the GPU; 99 means all
	MainGPU    int    // device index for a single-GPU box this is 0
	FlashAttn  bool   // enable flash attention
	CacheTypeK string // KV cache key type: "f16", "q8_0", "q4_0"
	CacheTypeV string // KV cache value type; must match CacheTypeK for the fused path
	Embedding  bool   // load in embedding mode (pooling on, causal attention off)

	// Speculative decoding, draft-model path. Ignored when DraftPath is empty.
	// The draft and target must share a vocabulary; a mismatch is detected at
	// load and fails loudly rather than silently corrupting output (doc 15).
	DraftMax  int     // max tokens drafted per verify pass (4 to 8 is the sweet spot)
	DraftMin  int     // min tokens drafted before a verify pass
	DraftPMin float32 // stop drafting when the draft token probability drops below this
}

// Sampling controls one generation. Zero values mean "use the engine default for
// this knob", which the adapter fills from the model's configured defaults before
// calling, so a request that omits temperature still gets a sensible value.
type Sampling struct {
	Temperature float32
	TopP        float32
	TopK        int
	MinP        float32
	Seed        uint32
	MaxTokens   int      // hard cap on generated tokens; 0 means "until EOG"
	Stop        []string // stop sequences; generation ends when one is produced
}

// Message is one chat turn. Role is "system", "user", or "assistant"; Content is
// the text. The engine applies the model's own chat template to a slice of these
// (llama_chat_apply_template) so the prompt formatting matches what the model was
// trained on, without the gateway hardcoding any template.
type Message struct {
	Role    string
	Content string
}

// Stats are the per-generation metrics the adapter logs and folds into the
// OpenAI usage object. DraftProposed and DraftAccepted are zero unless
// speculative decoding ran; their ratio is the acceptance rate from doc 15.
type Stats struct {
	PromptTokens     int
	CompletionTokens int
	DraftProposed    int
	DraftAccepted    int
	TTFT             time.Duration
	StopReason       string // "stop" (EOG or a stop sequence) or "length" (hit MaxTokens)
}

// Runner is a loaded model that generates in-process. A single Runner wraps one
// llama_context, which is single-stream, so the backend adapter serializes calls
// to one Runner with a mutex. Separate Runners (a main model and a coexisting
// embedding model) run concurrently.
type Runner interface {
	// Chat applies the model's chat template to msgs and generates a reply. It
	// calls emit for each new piece of decoded text as it is produced; if emit
	// returns false the generation stops early (the client disconnected). The
	// returned Stats carry the token counts, TTFT, and stop reason.
	Chat(ctx context.Context, msgs []Message, s Sampling, emit func(string) bool) (Stats, error)

	// Complete generates from a raw prompt with no chat template applied, the
	// /v1/completions path. The emit contract matches Chat.
	Complete(ctx context.Context, prompt string, s Sampling, emit func(string) bool) (Stats, error)

	// Embed returns the embedding vector for input. It is valid only on a Runner
	// loaded with Params.Embedding set; on a generative model it returns an error.
	Embed(ctx context.Context, input string) ([]float32, error)

	// Close frees the model, the context, and any draft model, releasing their
	// VRAM. After Close the Runner must not be used again.
	Close() error
}
