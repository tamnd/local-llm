package config

import (
	"strings"
	"testing"
)

const minimalYAML = `
bind:
  api_addr: "100.71.238.128:8888"
auth:
  tokens:
    - "sk-local-abc"
default_model: "daily"
models:
  qwen3-30b-a3b:
    backend: ollama
    base_url: "http://127.0.0.1:11434"
    upstream_model: "qwen3:30b-a3b"
    vram_mb: 20480
aliases:
  daily: qwen3-30b-a3b
`

func TestParseMinimalAppliesDefaults(t *testing.T) {
	cfg, err := Parse([]byte(minimalYAML))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if cfg.Bind.AdminAddr != "127.0.0.1:8889" {
		t.Errorf("admin_addr default = %q, want 127.0.0.1:8889", cfg.Bind.AdminAddr)
	}
	if cfg.Manager.QueueMax != 32 {
		t.Errorf("queue_max default = %d, want 32", cfg.Manager.QueueMax)
	}
	if cfg.Manager.VRAMBudgetMB != 22528 {
		t.Errorf("vram_budget_mb default = %d, want 22528", cfg.Manager.VRAMBudgetMB)
	}
	if cfg.Logging.Level != "info" || cfg.Logging.Format != "json" {
		t.Errorf("logging defaults = %q/%q, want info/json", cfg.Logging.Level, cfg.Logging.Format)
	}
}

func TestValidateRejects(t *testing.T) {
	cases := []struct {
		name string
		yaml string
		want string
	}{
		{
			name: "no api_addr",
			yaml: "auth:\n  tokens: [x]\ndefault_model: m\nmodels:\n  m: {backend: ollama, base_url: u, upstream_model: x}\n",
			want: "bind.api_addr is required",
		},
		{
			name: "no tokens",
			yaml: "bind:\n  api_addr: \"x:1\"\ndefault_model: m\nmodels:\n  m: {backend: ollama, base_url: u, upstream_model: x}\n",
			want: "auth.tokens needs at least one token",
		},
		{
			name: "unknown backend",
			yaml: "bind:\n  api_addr: \"x:1\"\nauth:\n  tokens: [t]\ndefault_model: m\nmodels:\n  m: {backend: bogus, base_url: u, upstream_model: x}\n",
			want: `unknown backend "bogus"`,
		},
		{
			name: "alias to alias",
			yaml: "bind:\n  api_addr: \"x:1\"\nauth:\n  tokens: [t]\ndefault_model: m\nmodels:\n  m: {backend: ollama, base_url: u, upstream_model: x}\naliases:\n  a: b\n  b: m\n",
			want: "points at another alias",
		},
		{
			name: "default model missing",
			yaml: "bind:\n  api_addr: \"x:1\"\nauth:\n  tokens: [t]\ndefault_model: ghost\nmodels:\n  m: {backend: ollama, base_url: u, upstream_model: x}\n",
			want: "default_model \"ghost\"",
		},
		{
			name: "unknown field",
			yaml: minimalYAML + "surprise: true\n",
			want: "parse config",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Parse([]byte(tc.yaml))
			if err == nil {
				t.Fatalf("Parse(%s): want error containing %q, got nil", tc.name, tc.want)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("Parse(%s): error = %q, want substring %q", tc.name, err, tc.want)
			}
		})
	}
}

func TestDefaultModelAliasResolves(t *testing.T) {
	cfg, err := Parse([]byte(minimalYAML))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if !cfg.resolves("daily") {
		t.Error("alias daily should resolve")
	}
	if cfg.resolves("nope") {
		t.Error("nonexistent name should not resolve")
	}
}
