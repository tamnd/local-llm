// Package observe is the only package that writes the request log to disk. Every
// other package routes its structured events through a Logger here, so swapping
// the observability backend later (Prometheus, a different sink) touches one
// place. The log is newline-delimited JSON: one object per line, the shape
// documented in spec 2065 doc 08 section 11.
package observe

import (
	"encoding/json"
	"io"
	"maps"
	"sync"
	"time"

	"github.com/tamnd/local-llm/config"
)

// levelRank orders log levels so the configured threshold can drop quieter
// events. A message is emitted when its rank is at least the threshold's rank.
var levelRank = map[string]int{
	"debug": 0,
	"info":  1,
	"warn":  2,
	"error": 3,
}

// Logger emits structured JSON events to an underlying writer (a rotating file
// in production, a buffer in tests). It is safe for concurrent use: the gateway
// logs one line per request from many goroutines at once.
type Logger struct {
	mu        sync.Mutex
	w         io.Writer
	threshold int
	now       func() time.Time
}

// New builds a Logger that writes to w, dropping events below cfg.Level. The
// caller owns w; for the on-disk log, pass the *RotatingWriter from NewFile.
func New(w io.Writer, level string) *Logger {
	rank, ok := levelRank[level]
	if !ok {
		rank = levelRank["info"]
	}
	return &Logger{w: w, threshold: rank, now: time.Now}
}

// Emit writes one JSON line for an event at the given level, merging the
// caller's fields with a timestamp, the level, and the event name. Fields below
// the configured threshold are dropped silently. A "ts", "level", or "event"
// key in fields is overwritten by the canonical values.
func (l *Logger) Emit(level, event string, fields map[string]any) {
	if levelRank[level] < l.threshold {
		return
	}
	rec := make(map[string]any, len(fields)+3)
	maps.Copy(rec, fields)
	rec["ts"] = l.now().UTC().Format(time.RFC3339Nano)
	rec["level"] = level
	rec["event"] = event

	line, err := json.Marshal(rec)
	if err != nil {
		// A field that will not marshal should never take down the gateway;
		// fall back to a minimal record that always encodes.
		line, _ = json.Marshal(map[string]any{
			"ts":    rec["ts"],
			"level": "error",
			"event": "log_marshal_failed",
			"orig":  event,
		})
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	// A failed write to the log sink must not take down the gateway; there is no
	// useful recovery for a broken log file mid-request, so the error is dropped.
	_, _ = l.w.Write(append(line, '\n'))
}

// Info emits an event at info level.
func (l *Logger) Info(event string, fields map[string]any) { l.Emit("info", event, fields) }

// Warn emits an event at warn level.
func (l *Logger) Warn(event string, fields map[string]any) { l.Emit("warn", event, fields) }

// Error emits an event at error level.
func (l *Logger) Error(event string, fields map[string]any) { l.Emit("error", event, fields) }

// Debug emits an event at debug level.
func (l *Logger) Debug(event string, fields map[string]any) { l.Emit("debug", event, fields) }

// Request is the per-request record from doc 08 section 11. It is a typed view
// of the fields the gateway logs after every completion so callers do not pass
// stringly-typed maps for the hot path.
type Request struct {
	RequestID        string
	Method           string
	Path             string
	Model            string
	Backend          string
	UpstreamModel    string
	TokenPrefix      string
	Status           int
	TTFTMillis       int64
	TotalMillis      int64
	PromptTokens     int
	CompletionTokens int
	Streamed         bool
}

// LogRequest emits a Request as an "info" event named "request".
func (l *Logger) LogRequest(r Request) {
	l.Info("request", map[string]any{
		"request_id":        r.RequestID,
		"method":            r.Method,
		"path":              r.Path,
		"model":             r.Model,
		"backend":           r.Backend,
		"upstream_model":    r.UpstreamModel,
		"token_prefix":      r.TokenPrefix,
		"status":            r.Status,
		"ttft_ms":           r.TTFTMillis,
		"total_ms":          r.TotalMillis,
		"prompt_tokens":     r.PromptTokens,
		"completion_tokens": r.CompletionTokens,
		"streamed":          r.Streamed,
	})
}

// FromConfig builds the production Logger and its writer from the logging
// config. When File is empty it logs to fallback (typically os.Stdout); when a
// path is set it opens a RotatingWriter. The returned io.Closer is the writer;
// close it on shutdown to flush and release the file handle.
func FromConfig(cfg config.Logging, fallback io.Writer) (*Logger, io.Closer, error) {
	if cfg.File == "" {
		return New(fallback, cfg.Level), nopCloser{}, nil
	}
	rw, err := NewFile(cfg.File, cfg.RotateMB, cfg.KeepFiles)
	if err != nil {
		return nil, nil, err
	}
	return New(rw, cfg.Level), rw, nil
}

type nopCloser struct{}

func (nopCloser) Close() error { return nil }
