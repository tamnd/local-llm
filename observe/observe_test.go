package observe

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tamnd/local-llm/config"
)

func TestEmitShapesLine(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf, "info")
	l.now = func() time.Time { return time.Unix(1750000000, 0).UTC() }

	l.Info("swap_start", map[string]any{"from_model": "a", "to_model": "b"})

	var rec map[string]any
	if err := json.Unmarshal(bytes.TrimSpace(buf.Bytes()), &rec); err != nil {
		t.Fatalf("emitted line is not JSON: %v", err)
	}
	if rec["event"] != "swap_start" || rec["level"] != "info" {
		t.Errorf("event/level = %v/%v, want swap_start/info", rec["event"], rec["level"])
	}
	if rec["from_model"] != "a" || rec["to_model"] != "b" {
		t.Errorf("fields not preserved: %v", rec)
	}
	if rec["ts"] != "2025-06-15T15:06:40Z" {
		t.Errorf("ts = %v, want fixed clock value", rec["ts"])
	}
}

func TestLevelThresholdDrops(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf, "warn")
	l.Info("quiet", nil)
	l.Debug("quieter", nil)
	if buf.Len() != 0 {
		t.Errorf("info/debug should be dropped at warn threshold, got %q", buf.String())
	}
	l.Error("loud", nil)
	if buf.Len() == 0 {
		t.Error("error should pass the warn threshold")
	}
}

func TestLogRequestFields(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf, "info")
	l.LogRequest(Request{
		RequestID: "req-1", Model: "qwen3-30b-a3b", Backend: "ollama",
		Status: 200, TTFTMillis: 142, Streamed: true,
	})
	var rec map[string]any
	if err := json.Unmarshal(bytes.TrimSpace(buf.Bytes()), &rec); err != nil {
		t.Fatalf("not JSON: %v", err)
	}
	if rec["ttft_ms"].(float64) != 142 || rec["streamed"] != true {
		t.Errorf("request fields wrong: %v", rec)
	}
}

func TestRotatingWriterRolls(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "llmgw.log")
	// Tiny threshold so a couple of writes force a rotation. maxBytes is in
	// MiB, so use the struct directly to set a byte-level threshold.
	w, err := NewFile(path, 0, 2)
	if err != nil {
		t.Fatalf("NewFile: %v", err)
	}
	w.maxBytes = 16
	defer w.Close()

	for range 5 {
		if _, err := w.Write([]byte("0123456789\n")); err != nil {
			t.Fatalf("Write: %v", err)
		}
	}
	if _, err := os.Stat(path + ".1"); err != nil {
		t.Errorf("expected rotated file %s.1: %v", path, err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Errorf("expected live file %s: %v", path, err)
	}
}

func TestFromConfigStdoutFallback(t *testing.T) {
	var buf bytes.Buffer
	l, closer, err := FromConfig(config.Logging{Level: "info"}, &buf)
	if err != nil {
		t.Fatalf("FromConfig: %v", err)
	}
	defer closer.Close()
	l.Info("hi", nil)
	if !strings.Contains(buf.String(), "\"event\":\"hi\"") {
		t.Errorf("fallback writer not used: %q", buf.String())
	}
}
