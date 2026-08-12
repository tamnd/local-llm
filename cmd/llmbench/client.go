package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// sample is one timed streaming generation: the wall-clock split into the
// time-to-first-token (prefill, compute-bound) and the decode phase
// (bandwidth-bound), plus the token counts the server reported in its usage
// block. Every derived rate in the bench comes from these five numbers, so the
// measurement lives in one place and is unit-tested against a canned stream.
type sample struct {
	ttft       time.Duration // request sent -> first content token
	decode     time.Duration // first content token -> last token
	promptToks int
	genToks    int
}

// prefillToksPerSec is prompt tokens divided by the prefill wall time. TabbyAPI
// and the OpenAI stream do not report a prompt_eval_duration, so prefill is
// measured as prompt_tokens / TTFT, the method spec 2065 doc 11 section 2.4
// fixes for the streaming backends.
func (s sample) prefillToksPerSec() float64 {
	if s.ttft <= 0 {
		return 0
	}
	return float64(s.promptToks) / s.ttft.Seconds()
}

// decodeToksPerSec is the headline number: generated tokens divided by the
// decode wall time, which excludes prefill so a long prompt does not drag the
// rate down (doc 11 section 2.3).
func (s sample) decodeToksPerSec() float64 {
	if s.decode <= 0 {
		return 0
	}
	return float64(s.genToks) / s.decode.Seconds()
}

// chatRequest is the minimal OpenAI chat-completions body the bench sends. It
// streams with usage included so the final chunk carries the token counts every
// backend the gateway fronts reports the same way.
type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	MaxTokens   int           `json:"max_tokens"`
	Temperature float64       `json:"temperature"`
	Stream      bool          `json:"stream"`
	StreamOpts  streamOptions `json:"stream_options"`
}

// chatMessage carries either a plain string (text-only cells) or the OpenAI
// content-part array a vision cell needs, so one request type covers both modes.
type chatMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// measure sends one streaming chat completion to base+"/chat/completions" and
// times it. It returns a sample or the first error; a non-2xx status is
// surfaced with the body so an OOM (503) or unreachable backend (502) reads
// clearly in the run log.
func measure(ctx context.Context, client *http.Client, base, token, model, prompt string, genToks int, now func() time.Time) (sample, error) {
	return measureContent(ctx, client, base, token, model, prompt, genToks, now)
}

// measureContent is measure's general form: content is either a string or a
// []contentPart carrying images. Vision cells go through here so the image
// encode cost lands inside TTFT exactly as a text prefill does, which is what
// makes the two modes comparable in the same schema.
func measureContent(ctx context.Context, client *http.Client, base, token, model string, content any, genToks int, now func() time.Time) (sample, error) {
	body := chatRequest{
		Model:       model,
		Messages:    []chatMessage{{Role: "user", Content: content}},
		MaxTokens:   genToks,
		Temperature: 0,
		Stream:      true,
		StreamOpts:  streamOptions{IncludeUsage: true},
	}
	buf, err := json.Marshal(body)
	if err != nil {
		return sample{}, err
	}
	url := strings.TrimRight(base, "/") + "/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(buf))
	if err != nil {
		return sample{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	start := now()
	resp, err := client.Do(req)
	if err != nil {
		return sample{}, fmt.Errorf("send: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode/100 != 2 {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return sample{}, fmt.Errorf("status %d: %s", resp.StatusCode, strings.TrimSpace(string(snippet)))
	}
	return readStream(resp.Body, start, now)
}

// readStream parses a text/event-stream of OpenAI chat chunks, stamping the
// first token that carries content as the TTFT boundary and the last event
// before [DONE] as the end. The usage block on the final chunk supplies the
// token counts. It is split out from measure so a canned stream can drive it in
// a test with no server.
func readStream(r io.Reader, start time.Time, now func() time.Time) (sample, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var s sample
	var firstAt, lastAt time.Time
	sawContent := false

	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			break
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
					// Reasoning models (qwen3, deepseek-r1) stream their chain of
					// thought in a separate channel: vLLM/DeepSeek use
					// reasoning_content, Ollama uses reasoning. Those tokens cost the
					// same decode work as content, so the rate must count them or a
					// thinking model that emits only reasoning within the token
					// budget reads as "no tokens" and the cell is wrongly skipped.
					ReasoningContent string `json:"reasoning_content"`
					Reasoning        string `json:"reasoning"`
				} `json:"delta"`
			} `json:"choices"`
			Usage *struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			// Tolerate keep-alive or comment frames the backend may interleave.
			continue
		}
		if chunk.Usage != nil {
			s.promptToks = chunk.Usage.PromptTokens
			s.genToks = chunk.Usage.CompletionTokens
		}
		hasToken := false
		if len(chunk.Choices) > 0 {
			d := chunk.Choices[0].Delta
			hasToken = d.Content != "" || d.ReasoningContent != "" || d.Reasoning != ""
		}
		if hasToken {
			t := now()
			if !sawContent {
				firstAt = t
				sawContent = true
			}
			lastAt = t
		}
	}
	if err := sc.Err(); err != nil {
		return sample{}, fmt.Errorf("read stream: %w", err)
	}
	if !sawContent {
		return sample{}, fmt.Errorf("stream carried no generated tokens")
	}
	s.ttft = firstAt.Sub(start)
	s.decode = lastAt.Sub(firstAt)
	return s, nil
}
