package backend

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"time"
)

// proxy is the shared HTTP forwarding core embedded by every adapter. It owns
// one http.Client (connection pooling across requests) and the gateway version
// string used in the system_fingerprint it injects. The proxy is what makes the
// adapters thin: forwarding and normalization live here once, not four times.
type proxy struct {
	client  *http.Client
	version string
}

// newProxy builds the shared proxy. The transport sets a 5 second dial timeout
// (the connect-timeout from doc 08 section 7.3); there is deliberately no
// client-wide timeout because streaming responses run arbitrarily long and are
// bounded by the request context instead.
func newProxy(version string) *proxy {
	return &proxy{
		version: version,
		client: &http.Client{
			Transport: &http.Transport{
				DialContext:           (&net.Dialer{Timeout: 5 * time.Second}).DialContext,
				MaxIdleConns:          16,
				IdleConnTimeout:       90 * time.Second,
				ResponseHeaderTimeout: 60 * time.Second, // first-byte timeout
			},
		},
	}
}

// forward sends req to baseURL+req.Path and writes the normalized response to w.
// backendID names the runtime for the system_fingerprint. It returns a Result
// with the metrics gathered while proxying.
func (p *proxy) forward(ctx context.Context, backendID, baseURL string, req *Request, w http.ResponseWriter) (*Result, error) {
	url := strings.TrimRight(baseURL, "/") + req.Path
	upstream, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(req.Body))
	if err != nil {
		return nil, fmt.Errorf("build upstream request: %w", err)
	}
	upstream.Header.Set("Content-Type", "application/json")
	if req.Stream {
		upstream.Header.Set("Accept", "text/event-stream")
	}

	start := time.Now()
	resp, err := p.client.Do(upstream)
	if err != nil {
		if errors.Is(err, context.Canceled) {
			return nil, context.Canceled
		}
		return nil, fmt.Errorf("%w: %w", ErrUnreachable, err)
	}
	defer func() { _ = resp.Body.Close() }()

	res := &Result{Status: resp.StatusCode, TTFT: time.Since(start)}
	fingerprint := fmt.Sprintf("llmgw-%s-%s", p.version, backendID)

	if req.Stream && resp.StatusCode == http.StatusOK {
		return res, p.streamSSE(resp.Body, w, fingerprint, res)
	}
	return res, p.copyJSON(resp, w, fingerprint, res)
}

// copyJSON handles the non-streaming path: read the whole body, normalize the
// finish_reason and usage fields, inject the system_fingerprint, and write it
// back with the upstream status code. Bodies that are not JSON (an error page,
// say) are passed through untouched.
func (p *proxy) copyJSON(resp *http.Response, w http.ResponseWriter, fingerprint string, res *Result) error {
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read upstream body: %w", err)
	}
	w.Header().Set("Content-Type", "application/json")

	var obj map[string]any
	if err := json.Unmarshal(body, &obj); err != nil {
		// Not JSON: pass through verbatim so the client sees the real error.
		w.WriteHeader(resp.StatusCode)
		_, werr := w.Write(body)
		return werr
	}
	normalizeObject(obj, fingerprint)
	recordUsage(obj, res)

	out, err := json.Marshal(obj)
	if err != nil {
		return fmt.Errorf("marshal normalized body: %w", err)
	}
	w.WriteHeader(resp.StatusCode)
	_, werr := w.Write(out)
	return werr
}

// streamSSE handles the streaming path: copy each SSE event from the backend to
// the client as it arrives, normalizing the JSON payload inline and flushing
// after every event so the client sees tokens in real time.
func (p *proxy) streamSSE(body io.Reader, w http.ResponseWriter, fingerprint string, res *Result) error {
	flusher, ok := w.(http.Flusher)
	if !ok {
		return errors.New("backend: response writer does not support streaming")
	}
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	reader := bufio.NewReader(body)
	for {
		line, err := reader.ReadBytes('\n')
		if len(line) > 0 {
			out := normalizeSSELine(line, fingerprint, res)
			if _, werr := w.Write(out); werr != nil {
				return werr
			}
			if bytes.HasPrefix(bytes.TrimSpace(line), []byte("data:")) {
				flusher.Flush()
			}
		}
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return fmt.Errorf("read sse stream: %w", err)
		}
	}
}

// normalizeSSELine rewrites one line of an SSE stream. Non-data lines and the
// terminal "[DONE]" sentinel pass through; a "data: {json}" line is parsed,
// normalized, and re-encoded. A payload that does not parse is forwarded as-is.
func normalizeSSELine(line []byte, fingerprint string, res *Result) []byte {
	trimmed := bytes.TrimRight(line, "\r\n")
	const prefix = "data:"
	if !bytes.HasPrefix(bytes.TrimSpace(trimmed), []byte(prefix)) {
		return line
	}
	payload := bytes.TrimSpace(trimmed[bytes.Index(trimmed, []byte(prefix))+len(prefix):])
	if bytes.Equal(payload, []byte("[DONE]")) {
		return line
	}
	var obj map[string]any
	if err := json.Unmarshal(payload, &obj); err != nil {
		return line
	}
	normalizeObject(obj, fingerprint)
	recordUsage(obj, res)
	out, err := json.Marshal(obj)
	if err != nil {
		return line
	}
	return append([]byte("data: "), append(out, '\n', '\n')...)
}

// finishReasonAliases maps the non-standard stop reasons some backends emit
// (older TabbyAPI builds in particular, doc 08 section 7.2) to the OpenAI set.
var finishReasonAliases = map[string]string{
	"eos":        "stop",
	"max_length": "length",
}

// normalizeObject applies the cross-backend normalizations to one response or
// chunk object in place: canonicalize finish_reason on every choice and inject
// the system_fingerprint when the backend did not set one.
func normalizeObject(obj map[string]any, fingerprint string) {
	if _, ok := obj["system_fingerprint"]; !ok {
		obj["system_fingerprint"] = fingerprint
	}
	choices, ok := obj["choices"].([]any)
	if !ok {
		return
	}
	for _, c := range choices {
		choice, ok := c.(map[string]any)
		if !ok {
			continue
		}
		if fr, ok := choice["finish_reason"].(string); ok {
			if canon, aliased := finishReasonAliases[fr]; aliased {
				choice["finish_reason"] = canon
			}
		}
	}
}

// recordUsage pulls prompt and completion token counts out of an object's usage
// field into the Result. Streaming backends put usage only on the final chunk,
// so this is called on every line and simply overwrites with the latest seen.
func recordUsage(obj map[string]any, res *Result) {
	usage, ok := obj["usage"].(map[string]any)
	if !ok {
		return
	}
	if v, ok := usage["prompt_tokens"].(float64); ok {
		res.PromptTokens = int(v)
	}
	if v, ok := usage["completion_tokens"].(float64); ok {
		res.CompletionTokens = int(v)
	}
}

// healthCheck is the shared Healthy implementation: a GET to baseURL+path with a
// short timeout. Any 2xx, or even a 404 from a runtime that lacks the probe
// path, proves the socket is alive; only a transport error means unreachable.
func (p *proxy) healthCheck(ctx context.Context, baseURL, path string) error {
	ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	url := strings.TrimRight(baseURL, "/") + path
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	resp, err := p.client.Do(req)
	if err != nil {
		return fmt.Errorf("%w: %w", ErrUnreachable, err)
	}
	_ = resp.Body.Close()
	return nil
}

// postJSON is a small helper for the API-driven Load/Unload calls (Ollama
// keep-alive, TabbyAPI load/unload). It posts body to baseURL+path and returns
// an error unless the response is 2xx; a CUDA-OOM message in the body is mapped
// to ErrOOM so the manager can surface a precise 503.
func (p *proxy) postJSON(ctx context.Context, baseURL, path string, body []byte, adminKey string) error {
	url := strings.TrimRight(baseURL, "/") + path
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	if adminKey != "" {
		req.Header.Set("Authorization", "Bearer "+adminKey)
	}
	resp, err := p.client.Do(req)
	if err != nil {
		return fmt.Errorf("%w: %w", ErrUnreachable, err)
	}
	defer func() { _ = resp.Body.Close() }()
	respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return nil
	}
	if isOOM(respBody) {
		return fmt.Errorf("%w (%s)", ErrOOM, path)
	}
	return fmt.Errorf("%s %s: status %d: %s", http.MethodPost, path, resp.StatusCode, strings.TrimSpace(string(respBody)))
}

// isOOM reports whether a backend error body looks like a CUDA out-of-memory.
func isOOM(body []byte) bool {
	low := strings.ToLower(string(body))
	return strings.Contains(low, "out of memory") || strings.Contains(low, "cuda oom") || strings.Contains(low, "outofmemory")
}
