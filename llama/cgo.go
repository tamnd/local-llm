//go:build llama

// This file is the real in-process engine, compiled only with `-tags llama` on a
// box that has a CUDA-linked libllama. It is deliberately thin: it loads a GGUF,
// tokenizes, applies the chat template, drives llama_decode, reads raw logits and
// embeddings back out, and frees everything on Close. Every decision that is easy
// to get subtly wrong (which token to pick, when a stop sequence has completed,
// how many speculative tokens to accept) lives in the pure-Go helpers in
// sampling.go and streamstop.go, which are unit-tested without a GPU. The cgo
// here only moves bytes across the boundary.
//
// The build expects libllama and its ggml backends on the linker path. See
// scripts/build-libllama.sh and the Makefile's build-llama target. Spec 2065
// doc 16 documents the toolchain.
package llama

/*
#cgo CFLAGS: -I${SRCDIR}/../third_party/llama.cpp/include -I${SRCDIR}/../third_party/llama.cpp/ggml/include
#cgo LDFLAGS: -L${SRCDIR}/../third_party/llama.cpp/build/bin -lllama -lggml -lggml-base -lggml-cpu -lggml-cuda -lstdc++ -lm
#include <stdlib.h>
#include <string.h>
#include "llama.h"
*/
import "C"

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"
	"unsafe"
)

// backendOnce guards the one-time global libllama init. llama_backend_init sets
// up the ggml backends (including CUDA) process-wide and must run before any
// model load; llama_backend_free is intentionally never called because the engine
// lives for the lifetime of the process.
var backendOnce sync.Once

func initBackend() { backendOnce.Do(func() { C.llama_backend_init() }) }

// Available reports that this binary was built with the in-process engine. The
// backend adapter checks it at startup before accepting an inproc model.
func Available() bool { return true }

// cgoRunner is one loaded model plus its context. When a draft model is loaded
// it also holds the draft handles for speculative decoding. A cgoRunner is
// single-stream; the backend adapter serializes calls to it with a mutex.
type cgoRunner struct {
	params Params

	model  *C.struct_llama_model
	ctx    *C.struct_llama_context
	vocab  *C.struct_llama_vocab
	nVocab int
	nEmbd  int
	nCtx   int

	draftModel *C.struct_llama_model
	draftCtx   *C.struct_llama_context
	draftVocab *C.struct_llama_vocab
}

// New loads a model into VRAM and returns a Runner. The tuning levers from
// Params (full GPU offload, flash attention, KV cache types) are applied here so
// the engine, not Ollama's defaults, controls them. A draft model, when given,
// is loaded into the same process for speculative decoding and is rejected if its
// vocabulary does not match the target, which is the silent-corruption failure
// mode doc 15 warns about.
func New(p Params) (Runner, error) {
	if p.ModelPath == "" {
		return nil, errors.New("llama: ModelPath is required")
	}
	initBackend()

	mparams := C.llama_model_default_params()
	if p.NGPULayers != 0 {
		mparams.n_gpu_layers = C.int32_t(p.NGPULayers)
	}
	mparams.main_gpu = C.int32_t(p.MainGPU)

	cpath := C.CString(p.ModelPath)
	defer C.free(unsafe.Pointer(cpath))
	model := C.llama_model_load_from_file(cpath, mparams)
	if model == nil {
		return nil, fmt.Errorf("llama: load model %q failed", p.ModelPath)
	}

	r := &cgoRunner{params: p, model: model}
	r.vocab = C.llama_model_get_vocab(model)
	r.nVocab = int(C.llama_vocab_n_tokens(r.vocab))
	r.nEmbd = int(C.llama_model_n_embd(model))

	cparams := C.llama_context_default_params()
	if p.NCtx > 0 {
		cparams.n_ctx = C.uint32_t(p.NCtx)
	}
	cparams.flash_attn = C.bool(p.FlashAttn)
	if k := cacheType(p.CacheTypeK); k != 0 {
		cparams.type_k = k
	}
	if v := cacheType(p.CacheTypeV); v != 0 {
		cparams.type_v = v
	}
	if p.Embedding {
		cparams.embeddings = C.bool(true)
		cparams.pooling_type = C.LLAMA_POOLING_TYPE_MEAN
	}

	r.ctx = C.llama_init_from_model(model, cparams)
	if r.ctx == nil {
		C.llama_model_free(model)
		return nil, errors.New("llama: create context failed")
	}
	r.nCtx = int(C.llama_n_ctx(r.ctx))

	if p.DraftPath != "" && !p.Embedding {
		if err := r.loadDraft(p, mparams); err != nil {
			r.Close()
			return nil, err
		}
	}
	return r, nil
}

// loadDraft loads the speculative draft model and verifies it shares the target
// vocabulary. A mismatch fails the load loudly rather than producing a draft
// context that silently corrupts the target's output.
func (r *cgoRunner) loadDraft(p Params, mparams C.struct_llama_model_params) error {
	cpath := C.CString(p.DraftPath)
	defer C.free(unsafe.Pointer(cpath))
	dm := C.llama_model_load_from_file(cpath, mparams)
	if dm == nil {
		return fmt.Errorf("llama: load draft model %q failed", p.DraftPath)
	}
	dv := C.llama_model_get_vocab(dm)
	if int(C.llama_vocab_n_tokens(dv)) != r.nVocab {
		C.llama_model_free(dm)
		return fmt.Errorf("llama: draft vocab size %d does not match target %d, refusing to speculate",
			int(C.llama_vocab_n_tokens(dv)), r.nVocab)
	}
	dparams := C.llama_context_default_params()
	dparams.n_ctx = C.uint32_t(r.nCtx)
	dparams.flash_attn = C.bool(p.FlashAttn)
	dctx := C.llama_init_from_model(dm, dparams)
	if dctx == nil {
		C.llama_model_free(dm)
		return errors.New("llama: create draft context failed")
	}
	r.draftModel = dm
	r.draftCtx = dctx
	r.draftVocab = dv
	return nil
}

// cacheType maps a config string to the ggml type enum for the KV cache. An empty
// or unknown string returns 0, which leaves the context default in place.
func cacheType(s string) C.enum_ggml_type {
	switch strings.ToLower(s) {
	case "f16":
		return C.GGML_TYPE_F16
	case "q8_0":
		return C.GGML_TYPE_Q8_0
	case "q4_0":
		return C.GGML_TYPE_Q4_0
	default:
		return 0
	}
}

// Chat applies the model's chat template to msgs and generates a reply.
func (r *cgoRunner) Chat(ctx context.Context, msgs []Message, s Sampling, emit func(string) bool) (Stats, error) {
	prompt, err := r.applyTemplate(msgs)
	if err != nil {
		return Stats{}, err
	}
	return r.generate(ctx, prompt, true, s, emit)
}

// Complete generates from a raw prompt with no template applied.
func (r *cgoRunner) Complete(ctx context.Context, prompt string, s Sampling, emit func(string) bool) (Stats, error) {
	return r.generate(ctx, prompt, false, s, emit)
}

// generate is the shared decode loop. It tokenizes the prompt, decodes it, then
// produces tokens until an end-of-generation token, a stop sequence, the token
// cap, or a cancelled context. When a draft model is loaded and sampling is
// greedy it runs the speculative path; otherwise it runs the standard path.
func (r *cgoRunner) generate(ctx context.Context, prompt string, addSpecial bool, s Sampling, emit func(string) bool) (Stats, error) {
	toks, err := r.tokenize(r.ctx, r.vocab, prompt, addSpecial)
	if err != nil {
		return Stats{}, err
	}
	if len(toks) >= r.nCtx {
		return Stats{}, ErrContextFull
	}

	smpl := newSampler(s)
	if r.draftCtx != nil && smpl.greedy() {
		return r.generateSpeculative(ctx, toks, s, emit)
	}
	return r.generateStandard(ctx, toks, smpl, s, emit)
}

// generateStandard is the single-model decode loop: decode the prompt, then
// sample one token per forward pass.
func (r *cgoRunner) generateStandard(ctx context.Context, prompt []C.llama_token, smpl *sampler, s Sampling, emit func(string) bool) (Stats, error) {
	start := time.Now()
	stats := Stats{PromptTokens: len(prompt)}
	stop := newStopFilter(s.Stop)

	r.clearMemory(r.ctx)
	if err := r.decode(r.ctx, prompt, 0, false); err != nil {
		return stats, err
	}
	pos := len(prompt)
	first := true

	for {
		if err := ctx.Err(); err != nil {
			return stats, err
		}
		if s.MaxTokens > 0 && stats.CompletionTokens >= s.MaxTokens {
			stats.StopReason = "length"
			break
		}
		tok := smpl.sample(r.logits(r.ctx, -1))
		if first {
			stats.TTFT = time.Since(start)
			first = false
		}
		if C.llama_vocab_is_eog(r.vocab, C.llama_token(tok)) {
			stats.StopReason = "stop"
			break
		}
		stats.CompletionTokens++
		out, stopped := stop.push(r.piece(C.llama_token(tok)))
		if out != "" && !emit(out) {
			stats.StopReason = "stop"
			return stats, nil
		}
		if stopped {
			stats.StopReason = "stop"
			return stats, nil
		}
		if err := r.decode(r.ctx, []C.llama_token{C.llama_token(tok)}, pos, false); err != nil {
			return stats, err
		}
		pos++
	}
	if tail := stop.flush(); tail != "" {
		emit(tail)
	}
	if stats.StopReason == "" {
		stats.StopReason = "stop"
	}
	return stats, nil
}

// generateSpeculative is the draft-and-verify loop. Each round the draft model
// proposes up to DraftMax tokens greedily, the target verifies them in one batch
// forward pass, and the longest prefix the target agrees with is accepted plus
// one bonus token of the target's own. The acceptance test and token bookkeeping
// are in the pure-Go acceptPrefix; the KV caches of both models are rolled back
// with llama_memory_seq_rm to the accepted length so the next round starts in
// sync. Speculative decoding only runs in greedy mode, where it is exact.
func (r *cgoRunner) generateSpeculative(ctx context.Context, prompt []C.llama_token, s Sampling, emit func(string) bool) (Stats, error) {
	start := time.Now()
	stats := Stats{PromptTokens: len(prompt)}
	stop := newStopFilter(s.Stop)

	r.clearMemory(r.ctx)
	r.clearMemory(r.draftCtx)
	if err := r.decode(r.ctx, prompt, 0, false); err != nil {
		return stats, err
	}
	if err := r.decode(r.draftCtx, prompt, 0, false); err != nil {
		return stats, err
	}
	pos := len(prompt)
	first := true

	draftMax := r.params.DraftMax
	if draftMax <= 0 {
		draftMax = 4
	}

	emitTok := func(tok int) (done bool, err error) {
		stats.CompletionTokens++
		out, stopped := stop.push(r.piece(C.llama_token(tok)))
		if out != "" && !emit(out) {
			return true, nil
		}
		return stopped, nil
	}

	for {
		if err := ctx.Err(); err != nil {
			return stats, err
		}
		if s.MaxTokens > 0 && stats.CompletionTokens >= s.MaxTokens {
			stats.StopReason = "length"
			break
		}

		// The target's current logits predict the token at pos. That is the first
		// slot the draft is trying to guess.
		targetFirst := argmax(r.logits(r.ctx, -1))
		if first {
			stats.TTFT = time.Since(start)
			first = false
		}

		// Draft proposes greedily, self-feeding, starting from the same prediction.
		draft := make([]int, 0, draftMax)
		next := targetFirst
		for i := 0; i < draftMax; i++ {
			draft = append(draft, next)
			if C.llama_vocab_is_eog(r.vocab, C.llama_token(next)) {
				break
			}
			if err := r.decode(r.draftCtx, []C.llama_token{C.llama_token(next)}, pos+i, false); err != nil {
				return stats, err
			}
			next = argmax(r.logits(r.draftCtx, -1))
		}
		stats.DraftProposed += len(draft)

		// Verify: decode the whole draft in the target with per-position logits.
		dtoks := make([]C.llama_token, len(draft))
		for i, t := range draft {
			dtoks[i] = C.llama_token(t)
		}
		if err := r.decode(r.ctx, dtoks, pos, true); err != nil {
			return stats, err
		}
		// target[i] is what the target itself would generate at draft slot i.
		// Slot 0 is the prediction we already had; slots 1.. come from the verify
		// pass logits at the preceding position.
		target := make([]int, len(draft))
		target[0] = targetFirst
		for i := 1; i < len(draft); i++ {
			target[i] = argmax(r.logits(r.ctx, i-1))
		}
		accepted := acceptPrefix(draft, target)
		// The bonus token is the target's own next token at the first divergence.
		// When the whole draft was accepted it is the target's prediction after the
		// last drafted position, read from that position's verify logits.
		var bonus int
		if accepted < len(draft) {
			bonus = target[accepted]
		} else {
			bonus = argmax(r.logits(r.ctx, len(draft)-1))
		}
		stats.DraftAccepted += accepted

		// Emit the accepted draft tokens, then the bonus token.
		stopHit := false
		for i := 0; i < accepted; i++ {
			done, err := emitTok(draft[i])
			if err != nil {
				return stats, err
			}
			if done {
				stopHit = true
				break
			}
		}
		if !stopHit && C.llama_vocab_is_eog(r.vocab, C.llama_token(bonus)) {
			stats.StopReason = "stop"
			if tail := stop.flush(); tail != "" {
				emit(tail)
			}
			return stats, nil
		}
		if !stopHit {
			done, err := emitTok(bonus)
			if err != nil {
				return stats, err
			}
			stopHit = done
		}
		if stopHit {
			stats.StopReason = "stop"
			return stats, nil
		}

		// Roll both caches back to the accepted prefix and seat the bonus token so
		// the next round starts with both models in sync at pos+accepted+1.
		newPos := pos + accepted
		r.memoryTrim(r.ctx, newPos)
		r.memoryTrim(r.draftCtx, newPos)
		if err := r.decode(r.ctx, []C.llama_token{C.llama_token(bonus)}, newPos, false); err != nil {
			return stats, err
		}
		if err := r.decode(r.draftCtx, []C.llama_token{C.llama_token(bonus)}, newPos, false); err != nil {
			return stats, err
		}
		pos = newPos + 1
	}
	if tail := stop.flush(); tail != "" {
		emit(tail)
	}
	if stats.StopReason == "" {
		stats.StopReason = "stop"
	}
	return stats, nil
}

// Embed returns the pooled embedding vector for input. It is valid only on a
// runner loaded with Embedding set.
func (r *cgoRunner) Embed(ctx context.Context, input string) ([]float32, error) {
	if !r.params.Embedding {
		return nil, errors.New("llama: runner not loaded for embeddings")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	toks, err := r.tokenize(r.ctx, r.vocab, input, true)
	if err != nil {
		return nil, err
	}
	r.clearMemory(r.ctx)
	if err := r.decode(r.ctx, toks, 0, false); err != nil {
		return nil, err
	}
	ptr := C.llama_get_embeddings_seq(r.ctx, 0)
	if ptr == nil {
		ptr = C.llama_get_embeddings_ith(r.ctx, C.int32_t(len(toks)-1))
	}
	if ptr == nil {
		return nil, errors.New("llama: no embedding produced")
	}
	src := unsafe.Slice((*float32)(unsafe.Pointer(ptr)), r.nEmbd)
	out := make([]float32, r.nEmbd)
	copy(out, src)
	return out, nil
}

// Close frees the context and model and any draft handles, releasing their VRAM.
func (r *cgoRunner) Close() error {
	if r.draftCtx != nil {
		C.llama_free(r.draftCtx)
		r.draftCtx = nil
	}
	if r.draftModel != nil {
		C.llama_model_free(r.draftModel)
		r.draftModel = nil
	}
	if r.ctx != nil {
		C.llama_free(r.ctx)
		r.ctx = nil
	}
	if r.model != nil {
		C.llama_model_free(r.model)
		r.model = nil
	}
	return nil
}

// tokenize converts text to token ids, sizing the buffer in two passes when the
// first guess is too small (llama_tokenize returns the negative required length).
func (r *cgoRunner) tokenize(_ *C.struct_llama_context, vocab *C.struct_llama_vocab, text string, addSpecial bool) ([]C.llama_token, error) {
	ctext := C.CString(text)
	defer C.free(unsafe.Pointer(ctext))
	n := C.int32_t(len(text) + 8)
	toks := make([]C.llama_token, int(n))
	got := C.llama_tokenize(vocab, ctext, C.int32_t(len(text)), &toks[0], n, C.bool(addSpecial), C.bool(true))
	if got < 0 {
		n = -got
		toks = make([]C.llama_token, int(n))
		got = C.llama_tokenize(vocab, ctext, C.int32_t(len(text)), &toks[0], n, C.bool(addSpecial), C.bool(true))
		if got < 0 {
			return nil, errors.New("llama: tokenize failed")
		}
	}
	if got == 0 {
		return nil, errors.New("llama: empty tokenization")
	}
	return toks[:int(got)], nil
}

// piece converts one token id to its text, sizing the buffer in two passes.
func (r *cgoRunner) piece(tok C.llama_token) string {
	buf := make([]C.char, 256)
	n := C.llama_token_to_piece(r.vocab, tok, &buf[0], C.int32_t(len(buf)), 0, C.bool(false))
	if n < 0 {
		buf = make([]C.char, int(-n))
		n = C.llama_token_to_piece(r.vocab, tok, &buf[0], C.int32_t(len(buf)), 0, C.bool(false))
	}
	if n <= 0 {
		return ""
	}
	return C.GoStringN(&buf[0], n)
}

// applyTemplate renders chat messages with the model's own chat template. When
// the model carries no template it falls back to a ChatML rendering so the engine
// still works rather than refusing the request.
func (r *cgoRunner) applyTemplate(msgs []Message) (string, error) {
	cmsgs := make([]C.struct_llama_chat_message, len(msgs))
	var frees []unsafe.Pointer
	defer func() {
		for _, p := range frees {
			C.free(p)
		}
	}()
	for i, m := range msgs {
		role := C.CString(m.Role)
		content := C.CString(m.Content)
		frees = append(frees, unsafe.Pointer(role), unsafe.Pointer(content))
		cmsgs[i].role = role
		cmsgs[i].content = content
	}

	tmpl := C.llama_model_chat_template(r.model, nil)
	if tmpl == nil {
		return chatML(msgs), nil
	}
	need := C.llama_chat_apply_template(tmpl, &cmsgs[0], C.size_t(len(msgs)), C.bool(true), nil, 0)
	if need <= 0 {
		return chatML(msgs), nil
	}
	buf := make([]C.char, int(need))
	got := C.llama_chat_apply_template(tmpl, &cmsgs[0], C.size_t(len(msgs)), C.bool(true), &buf[0], need)
	if got <= 0 {
		return chatML(msgs), nil
	}
	return C.GoStringN(&buf[0], got), nil
}

// decode runs one forward pass over toks placed at positions pos0.. in sequence
// 0. When logitsAll is set every position produces logits (needed by the
// speculative verify pass); otherwise only the last does.
func (r *cgoRunner) decode(cctx *C.struct_llama_context, toks []C.llama_token, pos0 int, logitsAll bool) error {
	n := len(toks)
	if n == 0 {
		return nil
	}
	b := C.llama_batch_init(C.int32_t(n), 0, 1)
	defer C.llama_batch_free(b)

	tokSlice := unsafe.Slice(b.token, n)
	posSlice := unsafe.Slice(b.pos, n)
	nSeqSlice := unsafe.Slice(b.n_seq_id, n)
	seqSlice := unsafe.Slice(b.seq_id, n)
	logitSlice := unsafe.Slice(b.logits, n)
	for i := 0; i < n; i++ {
		tokSlice[i] = toks[i]
		posSlice[i] = C.llama_pos(pos0 + i)
		nSeqSlice[i] = 1
		unsafe.Slice(seqSlice[i], 1)[0] = 0
		if logitsAll || i == n-1 {
			logitSlice[i] = 1
		} else {
			logitSlice[i] = 0
		}
	}
	b.n_tokens = C.int32_t(n)
	if rc := C.llama_decode(r0(cctx), b); rc != 0 {
		return fmt.Errorf("llama: decode failed (%d)", int(rc))
	}
	return nil
}

// r0 is a tiny helper so decode can take either the main or the draft context
// without the caller juggling pointer types.
func r0(c *C.struct_llama_context) *C.struct_llama_context { return c }

// logits returns the logits vector at batch index i (-1 for the last position).
func (r *cgoRunner) logits(cctx *C.struct_llama_context, i int) []float32 {
	ptr := C.llama_get_logits_ith(cctx, C.int32_t(i))
	return unsafe.Slice((*float32)(unsafe.Pointer(ptr)), r.nVocab)
}

// clearMemory drops the whole KV cache so the next prompt starts clean.
func (r *cgoRunner) clearMemory(cctx *C.struct_llama_context) {
	C.llama_memory_clear(C.llama_get_memory(cctx), C.bool(true))
}

// memoryTrim removes KV entries from position keep onward in sequence 0, the
// rollback the speculative loop uses after a partial accept.
func (r *cgoRunner) memoryTrim(cctx *C.struct_llama_context, keep int) {
	C.llama_memory_seq_rm(C.llama_get_memory(cctx), 0, C.llama_pos(keep), -1)
}

// chatML is the fallback prompt rendering for a model with no embedded template.
func chatML(msgs []Message) string {
	var b strings.Builder
	for _, m := range msgs {
		b.WriteString("<|im_start|>")
		b.WriteString(m.Role)
		b.WriteString("\n")
		b.WriteString(m.Content)
		b.WriteString("<|im_end|>\n")
	}
	b.WriteString("<|im_start|>assistant\n")
	return b.String()
}
