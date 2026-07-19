package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// fakeClock returns a now() that advances by step on each call, so a canned
// stream produces deterministic TTFT and decode durations without sleeping.
func fakeClock(start time.Time, step time.Duration) func() time.Time {
	t := start
	first := true
	return func() time.Time {
		if first {
			first = false
			return t
		}
		t = t.Add(step)
		return t
	}
}

func TestMeasureTimesPrefillAndDecode(t *testing.T) {
	// A stream where the first data frame carries no content (role only), then
	// three content tokens, then a usage-only final frame, then [DONE].
	sse := "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"one\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\" two\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\" three\"}}]}\n\n" +
		"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":3}}\n\n" +
		"data: [DONE]\n\n"

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer tok" {
			t.Errorf("missing bearer token, got %q", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(sse))
	}))
	defer srv.Close()

	// Clock: start at T0, each now() call adds 1s. Calls happen at: request
	// start (T0), then once per content token as they arrive (T1, T2, T3).
	start := time.Unix(0, 0)
	now := fakeClock(start, time.Second)

	s, err := measure(context.Background(), srv.Client(), srv.URL+"/v1", "tok", "m", "hi", 8, now)
	if err != nil {
		t.Fatalf("measure: %v", err)
	}
	if s.promptToks != 100 || s.genToks != 3 {
		t.Fatalf("token counts: prompt=%d gen=%d, want 100/3", s.promptToks, s.genToks)
	}
	// start=T0, first content at T1 -> ttft 1s; last content at T3 -> decode 2s.
	if s.ttft != time.Second {
		t.Errorf("ttft = %v, want 1s", s.ttft)
	}
	if s.decode != 2*time.Second {
		t.Errorf("decode = %v, want 2s", s.decode)
	}
	if got := s.decodeToksPerSec(); got != 1.5 {
		t.Errorf("decode tok/s = %v, want 1.5 (3 toks / 2s)", got)
	}
	if got := s.prefillToksPerSec(); got != 100 {
		t.Errorf("prefill tok/s = %v, want 100 (100 toks / 1s)", got)
	}
}

func TestMeasureSurfacesNon2xx(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte("out of memory"))
	}))
	defer srv.Close()

	_, err := measure(context.Background(), srv.Client(), srv.URL+"/v1", "", "m", "hi", 8, time.Now)
	if err == nil {
		t.Fatal("want error on 503, got nil")
	}
}

func TestReadStreamNoContentIsError(t *testing.T) {
	sse := "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\ndata: [DONE]\n\n"
	_, err := readStream(strings.NewReader(sse), time.Unix(0, 0), fakeClock(time.Unix(0, 0), time.Second))
	if err == nil {
		t.Fatal("want error when stream carries no content tokens")
	}
}

func TestMeanStd(t *testing.T) {
	mean, std := meanStd([]float64{10, 12, 14})
	if mean != 12 {
		t.Errorf("mean = %v, want 12", mean)
	}
	if std < 1.9 || std > 2.1 {
		t.Errorf("std = %v, want ~2", std)
	}
}
