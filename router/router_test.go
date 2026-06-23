package router

import (
	"reflect"
	"testing"

	"github.com/tamnd/local-llm/config"
)

func testRouter() *Router {
	return New(&config.Config{
		DefaultModel: "default",
		Models: map[string]config.ModelEntry{
			"qwen3-30b-a3b":       {Backend: "ollama", BaseURL: "u", UpstreamModel: "qwen3:30b-a3b"},
			"qwen3-coder-30b-a3b": {Backend: "ollama", BaseURL: "u", UpstreamModel: "qwen3-coder:30b-a3b"},
		},
		Aliases: map[string]string{
			"default": "qwen3-30b-a3b",
			"coder":   "qwen3-coder-30b-a3b",
		},
	})
}

func TestResolveDirectAndAlias(t *testing.T) {
	r := testRouter()
	cases := []struct {
		in   string
		want string
	}{
		{"qwen3-30b-a3b", "qwen3-30b-a3b"}, // direct id
		{"coder", "qwen3-coder-30b-a3b"},   // alias
		{"", "qwen3-30b-a3b"},              // empty -> default alias -> model
		{"default", "qwen3-30b-a3b"},       // default alias by name
	}
	for _, tc := range cases {
		got, ok := r.Resolve(tc.in)
		if !ok {
			t.Errorf("Resolve(%q) not found", tc.in)
			continue
		}
		if got.ID != tc.want {
			t.Errorf("Resolve(%q).ID = %q, want %q", tc.in, got.ID, tc.want)
		}
	}
}

func TestResolveUnknown(t *testing.T) {
	r := testRouter()
	if _, ok := r.Resolve("gpt-4o"); ok {
		t.Error("unknown model should not resolve")
	}
}

func TestIDsSortedNoAliases(t *testing.T) {
	r := testRouter()
	got := r.IDs()
	want := []string{"qwen3-30b-a3b", "qwen3-coder-30b-a3b"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("IDs() = %v, want %v", got, want)
	}
}
