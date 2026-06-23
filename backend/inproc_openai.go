package backend

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/tamnd/local-llm/config"
	"github.com/tamnd/local-llm/llama"
)

// This file turns OpenAI-shaped requests into llama engine calls and the engine's
// output back into OpenAI-shaped JSON or SSE. The other backends get this framing
// for free by proxying a runtime that already speaks OpenAI; the in-process
// engine produces tokens, so the gateway has to assemble the wire format itself.
// The shapes here match what the proxy normalizes for the HTTP backends:
// chat.completion / chat.completion.chunk, a system_fingerprint of
// llmgw-<version>-inproc, finish_reason of "stop" or "length", and a terminal
// "data: [DONE]" on streams.

// chatRequest is the subset of the OpenAI chat body the engine reads. Sampling
// fields are pointers so an omitted field falls back to the model default rather
// than to a zero that would force greedy decoding.
type chatRequest struct {
	Messages    []chatMessage `json:"messages"`
	Temperature *float64      `json:"temperature"`
	TopP        *float64      `json:"top_p"`
	TopK        *int          `json:"top_k"`
	MinP        *float64      `json:"min_p"`
	MaxTokens   *int          `json:"max_tokens"`
	Seed        *uint32       `json:"seed"`
	Stop        stopValue     `json:"stop"`
	Stream      bool          `json:"stream"`
}

// chatMessage carries one turn. Content is raw because OpenAI allows either a
// string or an array of typed parts; contentText flattens both.
type chatMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// completionRequest is the /v1/completions body.
type completionRequest struct {
	Prompt      json.RawMessage `json:"prompt"`
	Temperature *float64        `json:"temperature"`
	TopP        *float64        `json:"top_p"`
	TopK        *int            `json:"top_k"`
	MinP        *float64        `json:"min_p"`
	MaxTokens   *int            `json:"max_tokens"`
	Seed        *uint32         `json:"seed"`
	Stop        stopValue       `json:"stop"`
	Stream      bool            `json:"stream"`
}

// embeddingRequest is the /v1/embeddings body. Input is raw because it may be a
// string or an array of strings.
type embeddingRequest struct {
	Input json.RawMessage `json:"input"`
}

// stopValue decodes the OpenAI "stop" field, which is either a single string or
// an array of strings.
type stopValue []string

func (s *stopValue) UnmarshalJSON(b []byte) error {
	if len(b) == 0 || string(b) == "null" {
		return nil
	}
	if b[0] == '"' {
		var one string
		if err := json.Unmarshal(b, &one); err != nil {
			return err
		}
		*s = []string{one}
		return nil
	}
	var many []string
	if err := json.Unmarshal(b, &many); err != nil {
		return err
	}
	*s = many
	return nil
}

// samplingFrom merges request sampling fields over the model's configured
// defaults, so a request that omits temperature still gets the model's default
// rather than greedy decoding.
func samplingFrom(entry config.ModelEntry, temp, topP, minP *float64, topK *int, maxTokens *int, seed *uint32, stop []string) llama.Sampling {
	s := llama.Sampling{
		Temperature: paramFloat32(entry.Params, "temperature", 0.7),
		TopP:        paramFloat32(entry.Params, "top_p", 0.95),
		TopK:        float32ToInt(paramFloat32(entry.Params, "top_k", 0)),
		MinP:        paramFloat32(entry.Params, "min_p", 0),
		Stop:        stop,
	}
	if temp != nil {
		s.Temperature = float32(*temp)
	}
	if topP != nil {
		s.TopP = float32(*topP)
	}
	if minP != nil {
		s.MinP = float32(*minP)
	}
	if topK != nil {
		s.TopK = *topK
	}
	if maxTokens != nil {
		s.MaxTokens = *maxTokens
	}
	if seed != nil {
		s.Seed = *seed
	}
	return s
}

func float32ToInt(f float32) int { return int(f) }

// serveChat runs a chat request. For a streaming request it emits
// chat.completion.chunk events as tokens arrive; otherwise it accumulates the
// full reply and writes one chat.completion object.
func serveChat(ctx context.Context, runner llama.Runner, entry config.ModelEntry, req *Request, w http.ResponseWriter, fingerprint string) (*Result, error) {
	var body chatRequest
	if err := json.Unmarshal(req.Body, &body); err != nil {
		return nil, fmt.Errorf("inproc: decode chat request: %w", err)
	}
	msgs := make([]llama.Message, 0, len(body.Messages))
	for _, m := range body.Messages {
		msgs = append(msgs, llama.Message{Role: m.Role, Content: contentText(m.Content)})
	}
	s := samplingFrom(entry, body.Temperature, body.TopP, body.MinP, body.TopK, body.MaxTokens, body.Seed, body.Stop)
	model := entry.UpstreamModel

	gen := func(emit func(string) bool) (llama.Stats, error) {
		return runner.Chat(ctx, msgs, s, emit)
	}
	if req.Stream {
		return streamChat(gen, w, model, fingerprint, "chat.completion.chunk", true)
	}
	return blockChat(gen, w, model, fingerprint)
}

// serveCompletion runs a /v1/completions request with no chat template.
func serveCompletion(ctx context.Context, runner llama.Runner, entry config.ModelEntry, req *Request, w http.ResponseWriter, fingerprint string) (*Result, error) {
	var body completionRequest
	if err := json.Unmarshal(req.Body, &body); err != nil {
		return nil, fmt.Errorf("inproc: decode completion request: %w", err)
	}
	prompt := firstString(body.Prompt)
	s := samplingFrom(entry, body.Temperature, body.TopP, body.MinP, body.TopK, body.MaxTokens, body.Seed, body.Stop)
	model := entry.UpstreamModel

	gen := func(emit func(string) bool) (llama.Stats, error) {
		return runner.Complete(ctx, prompt, s, emit)
	}
	if req.Stream {
		return streamChat(gen, w, model, fingerprint, "text_completion", false)
	}
	return blockCompletion(gen, w, model, fingerprint)
}

// serveEmbeddings runs a /v1/embeddings request, returning one vector per input.
func serveEmbeddings(ctx context.Context, runner llama.Runner, entry config.ModelEntry, req *Request, w http.ResponseWriter, fingerprint string) (*Result, error) {
	var body embeddingRequest
	if err := json.Unmarshal(req.Body, &body); err != nil {
		return nil, fmt.Errorf("inproc: decode embedding request: %w", err)
	}
	inputs := stringList(body.Input)
	data := make([]map[string]any, 0, len(inputs))
	var promptTokens int
	for idx, in := range inputs {
		vec, err := runner.Embed(ctx, in)
		if err != nil {
			return nil, err
		}
		promptTokens += len(strings.Fields(in))
		data = append(data, map[string]any{
			"object":    "embedding",
			"index":     idx,
			"embedding": vec,
		})
	}
	out := map[string]any{
		"object":             "list",
		"data":               data,
		"model":              entry.UpstreamModel,
		"system_fingerprint": fingerprint,
		"usage": map[string]any{
			"prompt_tokens": promptTokens,
			"total_tokens":  promptTokens,
		},
	}
	res := &Result{Status: http.StatusOK, PromptTokens: promptTokens}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	return res, json.NewEncoder(w).Encode(out)
}

// blockChat runs a non-streaming chat generation and writes one chat.completion.
func blockChat(gen func(func(string) bool) (llama.Stats, error), w http.ResponseWriter, model, fingerprint string) (*Result, error) {
	var sb strings.Builder
	stats, err := gen(func(piece string) bool {
		sb.WriteString(piece)
		return true
	})
	if err != nil && !errors.Is(err, context.Canceled) {
		return nil, err
	}
	id := completionID("chatcmpl")
	obj := map[string]any{
		"id":                 id,
		"object":             "chat.completion",
		"created":            time.Now().Unix(),
		"model":              model,
		"system_fingerprint": fingerprint,
		"choices": []map[string]any{{
			"index":         0,
			"message":       map[string]any{"role": "assistant", "content": sb.String()},
			"finish_reason": stats.StopReason,
		}},
		"usage": usageObject(stats),
	}
	res := resultFrom(stats)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	return res, json.NewEncoder(w).Encode(obj)
}

// blockCompletion runs a non-streaming completion and writes one text_completion.
func blockCompletion(gen func(func(string) bool) (llama.Stats, error), w http.ResponseWriter, model, fingerprint string) (*Result, error) {
	var sb strings.Builder
	stats, err := gen(func(piece string) bool {
		sb.WriteString(piece)
		return true
	})
	if err != nil && !errors.Is(err, context.Canceled) {
		return nil, err
	}
	obj := map[string]any{
		"id":                 completionID("cmpl"),
		"object":             "text_completion",
		"created":            time.Now().Unix(),
		"model":              model,
		"system_fingerprint": fingerprint,
		"choices": []map[string]any{{
			"index":         0,
			"text":          sb.String(),
			"finish_reason": stats.StopReason,
		}},
		"usage": usageObject(stats),
	}
	res := resultFrom(stats)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	return res, json.NewEncoder(w).Encode(obj)
}

// streamChat runs a streaming generation, writing one SSE event per emitted
// piece and a terminal [DONE]. chunkObject is "chat.completion.chunk" for chat
// and "text_completion" for completions; chatShape selects the delta vs text
// field. It returns when generation ends or the client write fails.
func streamChat(gen func(func(string) bool) (llama.Stats, error), w http.ResponseWriter, model, fingerprint, chunkObject string, chatShape bool) (*Result, error) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		return nil, errors.New("inproc: response writer does not support streaming")
	}
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	id := completionID("chatcmpl")
	created := time.Now().Unix()
	first := true
	writeErr := false

	writeChunk := func(choice map[string]any) bool {
		obj := map[string]any{
			"id":                 id,
			"object":             chunkObject,
			"created":            created,
			"model":              model,
			"system_fingerprint": fingerprint,
			"choices":            []map[string]any{choice},
		}
		payload, _ := json.Marshal(obj)
		if _, err := fmt.Fprintf(w, "data: %s\n\n", payload); err != nil {
			writeErr = true
			return false
		}
		flusher.Flush()
		return true
	}

	stats, genErr := gen(func(piece string) bool {
		if piece == "" {
			return true
		}
		var choice map[string]any
		if chatShape {
			delta := map[string]any{"content": piece}
			if first {
				delta["role"] = "assistant"
			}
			choice = map[string]any{"index": 0, "delta": delta, "finish_reason": nil}
		} else {
			choice = map[string]any{"index": 0, "text": piece, "finish_reason": nil}
		}
		first = false
		return writeChunk(choice)
	})
	if genErr != nil && !errors.Is(genErr, context.Canceled) {
		// The stream has already started, so the status is sent; the best we can do
		// is stop and let the client see a truncated stream. Surface for logging.
		return resultFrom(stats), genErr
	}
	if writeErr {
		return resultFrom(stats), nil
	}

	// Final chunk: empty delta, the finish reason, and usage.
	var finalChoice map[string]any
	if chatShape {
		finalChoice = map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": stats.StopReason}
	} else {
		finalChoice = map[string]any{"index": 0, "text": "", "finish_reason": stats.StopReason}
	}
	finalObj := map[string]any{
		"id":                 id,
		"object":             chunkObject,
		"created":            created,
		"model":              model,
		"system_fingerprint": fingerprint,
		"choices":            []map[string]any{finalChoice},
		"usage":              usageObject(stats),
	}
	payload, _ := json.Marshal(finalObj)
	_, _ = fmt.Fprintf(w, "data: %s\n\n", payload)
	_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
	return resultFrom(stats), nil
}

// contentText flattens an OpenAI message content field, which is either a string
// or an array of typed parts, into plain text. Non-text parts (images) are
// dropped; the in-process engine path is text-only.
func contentText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	if raw[0] == '"' {
		var s string
		if json.Unmarshal(raw, &s) == nil {
			return s
		}
		return ""
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &parts) != nil {
		return ""
	}
	var b strings.Builder
	for _, p := range parts {
		if p.Type == "text" || p.Text != "" {
			b.WriteString(p.Text)
		}
	}
	return b.String()
}

// firstString returns the first string in a prompt field that is either a string
// or an array of strings.
func firstString(raw json.RawMessage) string {
	list := stringList(raw)
	if len(list) == 0 {
		return ""
	}
	return list[0]
}

// stringList decodes a field that is a string or an array of strings into a
// slice. An empty or null field yields an empty slice.
func stringList(raw json.RawMessage) []string {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	if raw[0] == '"' {
		var s string
		if json.Unmarshal(raw, &s) == nil {
			return []string{s}
		}
		return nil
	}
	var many []string
	if json.Unmarshal(raw, &many) == nil {
		return many
	}
	return nil
}

// usageObject builds the OpenAI usage object from engine stats.
func usageObject(s llama.Stats) map[string]any {
	return map[string]any{
		"prompt_tokens":     s.PromptTokens,
		"completion_tokens": s.CompletionTokens,
		"total_tokens":      s.PromptTokens + s.CompletionTokens,
	}
}

// resultFrom maps engine stats into the Result the gateway logs.
func resultFrom(s llama.Stats) *Result {
	return &Result{
		Status:           http.StatusOK,
		TTFT:             s.TTFT,
		PromptTokens:     s.PromptTokens,
		CompletionTokens: s.CompletionTokens,
	}
}

// completionID builds a response id. The nanosecond clock is enough to keep ids
// distinct within a single-user gateway without pulling in a UUID dependency.
func completionID(prefix string) string {
	return fmt.Sprintf("%s-%d", prefix, time.Now().UnixNano())
}
