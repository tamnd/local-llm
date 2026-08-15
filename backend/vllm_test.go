package backend

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tamnd/local-llm/config"
)

// TestVLLMLoadAdoptsHealthyServer covers the WSL/systemd path: when a vLLM server
// is already answering /health at base_url, Load adopts it and never spawns a
// process. This is what lets the Windows-host gateway serve a vLLM that runs in
// WSL2, where it cannot exec the Linux binary itself.
func TestVLLMLoadAdoptsHealthyServer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	fl := &fakeLauncher{}
	v := &VLLM{proxy: newProxy("test"), procs: &procTable{launcher: fl}}
	entry := config.ModelEntry{
		BaseURL:       srv.URL,
		UpstreamModel: "/models/hf/Qwen3-32B-AWQ",
		Params:        map[string]any{"quantization": "awq_marlin"},
	}
	if err := v.Load(context.Background(), entry); err != nil {
		t.Fatalf("Load of a healthy server should succeed by adoption, got %v", err)
	}
	if len(fl.started) != 0 {
		t.Errorf("expected no spawn when the server is already healthy, launched %v", fl.started)
	}
}

// hasFlagValue reports whether args contains flag immediately followed by value.
func hasFlagValue(args []string, flag, value string) bool {
	for i := 0; i < len(args)-1; i++ {
		if args[i] == flag && args[i+1] == value {
			return true
		}
	}
	return false
}

func hasFlag(args []string, flag string) bool {
	for _, a := range args {
		if a == flag {
			return true
		}
	}
	return false
}

// TestBuildVLLMArgsAWQ covers the dense 4-bit bench arm: the Marlin kernel, the
// 32k context, and the fp8 KV cache that lets it fit the 24 GB card. The fp8 KV
// flag is the one that used to be dropped, so a 32k fp16 cache overflowed VRAM.
func TestBuildVLLMArgsAWQ(t *testing.T) {
	entry := config.ModelEntry{
		BaseURL:       "http://127.0.0.1:8102",
		UpstreamModel: "/models/hf/Qwen3-32B-AWQ",
		Params: map[string]any{
			"quantization":   "awq_marlin",
			"kv_cache_dtype": "fp8",
			"max_model_len":  32768,
		},
	}
	args := buildVLLMArgs(entry)
	if args[0] != "serve" || args[1] != "/models/hf/Qwen3-32B-AWQ" {
		t.Fatalf("expected `serve <model>` first, got %v", args[:2])
	}
	if !hasFlagValue(args, "--port", "8102") {
		t.Errorf("port not derived from base_url: %v", args)
	}
	if !hasFlagValue(args, "--quantization", "awq_marlin") {
		t.Errorf("quantization not passed: %v", args)
	}
	if !hasFlagValue(args, "--kv-cache-dtype", "fp8") {
		t.Errorf("kv-cache-dtype fp8 not passed: %v", args)
	}
	if !hasFlagValue(args, "--max-model-len", "32768") {
		t.Errorf("max-model-len not passed: %v", args)
	}
}

// TestBuildVLLMArgsNoKVCacheDtype confirms the fp8 flag is opt-in: a model that
// does not set kv_cache_dtype keeps vLLM's default fp16 cache, so the flag must
// be absent rather than emitted empty.
//
// --quantization is opt-in for the same reason. It used to default to awq,
// which was invisible only because every vLLM entry happened to be AWQ; a
// checkpoint that carries its own scheme in config.json (native FP8, MXFP4)
// must not have awq forced onto it. This asserts the flag is now absent unless
// asked for; TestBuildVLLMArgs covers the case where it is set.
func TestBuildVLLMArgsNoKVCacheDtype(t *testing.T) {
	entry := config.ModelEntry{
		BaseURL:       "http://127.0.0.1:8100",
		UpstreamModel: "/models/hf/Qwen3-14B-FP8",
		Params:        map[string]any{},
	}
	args := buildVLLMArgs(entry)
	if hasFlag(args, "--kv-cache-dtype") {
		t.Errorf("kv-cache-dtype should be absent without the param: %v", args)
	}
	if strings.Contains(strings.Join(args, " "), "--quantization") {
		t.Errorf("quantization should be absent without the param: %v", args)
	}
	// Sanity: the builder is not a no-op.
	if !hasFlagValue(args, "--port", "8100") {
		t.Errorf("expected the port to be derived from base_url: %v", args)
	}
}
