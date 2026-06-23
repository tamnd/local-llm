// Package config holds the gateway configuration schema and the loader that
// reads it from YAML. It is the leaf package every other package depends on:
// the shared types (Config, ModelEntry, and friends) live here so the router,
// the manager, and the backend adapters can all refer to the same structs
// without an import cycle.
//
// The schema mirrors the YAML documented in spec 2065 doc 08 section 6. Any
// field that is not required has a default applied in applyDefaults, so a
// minimal config that names a bind address, one token, and one model is enough
// to start the gateway.
package config

import (
	"fmt"
	"net"
	"os"
	"strings"

	"gopkg.in/yaml.v3"
)

// Config is the full gateway configuration. It is parsed from a single YAML
// file and validated once at startup; a missing required field is a fatal
// error before the HTTP servers bind, never a surprise at request time.
type Config struct {
	Bind         Bind                  `yaml:"bind"`
	Auth         Auth                  `yaml:"auth"`
	DefaultModel string                `yaml:"default_model"`
	Models       map[string]ModelEntry `yaml:"models"`
	Aliases      map[string]string     `yaml:"aliases"`
	Manager      Manager               `yaml:"manager"`
	Logging      Logging               `yaml:"logging"`
}

// Bind controls the two listening sockets. The public API binds to the
// Tailscale interface; the admin surface binds to loopback only so that
// force-load and force-unload can never be reached from the network. See doc 08
// section 8.3.
type Bind struct {
	APIAddr   string `yaml:"api_addr"`
	AdminAddr string `yaml:"admin_addr"`
}

// Auth is the bearer-token policy. Tokens authenticate the public API; the
// admin token guards the loopback admin endpoints. If AdminToken is empty the
// admin endpoints are disabled entirely.
type Auth struct {
	Tokens     []string `yaml:"tokens"`
	AdminToken string   `yaml:"admin_token"`
}

// ModelEntry describes one model the gateway can serve: which backend owns it,
// where that backend listens, what the backend calls the model, and the VRAM it
// is expected to occupy. The manager uses VRAMMB for the coexistence check; the
// backend adapters use UpstreamModel and Params when they forward a request.
type ModelEntry struct {
	Backend       string         `yaml:"backend"`
	BaseURL       string         `yaml:"base_url"`
	UpstreamModel string         `yaml:"upstream_model"`
	VRAMMB        int            `yaml:"vram_mb"`
	Params        map[string]any `yaml:"params"`
	Coexist       bool           `yaml:"coexist"`
}

// Manager is the swap and queue policy. The timeouts bound how long a load,
// unload, or drain may take; VRAMBudgetMB is the ceiling the coexistence check
// measures against (the physical 22 GB minus a safety margin).
type Manager struct {
	DrainTimeoutS  int  `yaml:"drain_timeout_s"`
	QueueMax       int  `yaml:"queue_max"`
	LoadTimeoutS   int  `yaml:"load_timeout_s"`
	UnloadTimeoutS int  `yaml:"unload_timeout_s"`
	VRAMBudgetMB   int  `yaml:"vram_budget_mb"`
	HotSwap        bool `yaml:"hot_swap"`
}

// Logging configures the structured request log: where it goes, how it is
// formatted, and when it rotates.
type Logging struct {
	Level     string `yaml:"level"`
	Format    string `yaml:"format"`
	File      string `yaml:"file"`
	RotateMB  int    `yaml:"rotate_mb"`
	KeepFiles int    `yaml:"keep_files"`
}

// Known backend ids. A model entry's Backend must be one of these.
const (
	BackendOllama = "ollama"
	BackendLlama  = "llama"
	BackendTabby  = "tabby"
	BackendVLLM   = "vllm"
)

// validBackends is the set of backend ids the gateway knows how to talk to.
var validBackends = map[string]bool{
	BackendOllama: true,
	BackendLlama:  true,
	BackendTabby:  true,
	BackendVLLM:   true,
}

// Load reads a YAML config from path, applies defaults, and validates it. It
// returns a usable Config or an error that names the first problem found.
func Load(path string) (*Config, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config %s: %w", path, err)
	}
	return Parse(raw)
}

// Parse decodes a YAML document, applies defaults, and validates. It is split
// out from Load so tests can feed bytes without touching the filesystem.
func Parse(raw []byte) (*Config, error) {
	var cfg Config
	dec := yaml.NewDecoder(strings.NewReader(string(raw)))
	dec.KnownFields(true)
	if err := dec.Decode(&cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}
	cfg.applyDefaults()
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &cfg, nil
}

// applyDefaults fills in every optional field that was left blank. The defaults
// are the ones argued in doc 08 section 6.2 and doc 14 section 2.4.
func (c *Config) applyDefaults() {
	if c.Bind.AdminAddr == "" {
		c.Bind.AdminAddr = "127.0.0.1:8889"
	}
	if c.Manager.DrainTimeoutS == 0 {
		c.Manager.DrainTimeoutS = 120
	}
	if c.Manager.QueueMax == 0 {
		c.Manager.QueueMax = 32
	}
	if c.Manager.LoadTimeoutS == 0 {
		c.Manager.LoadTimeoutS = 120
	}
	if c.Manager.UnloadTimeoutS == 0 {
		c.Manager.UnloadTimeoutS = 30
	}
	if c.Manager.VRAMBudgetMB == 0 {
		c.Manager.VRAMBudgetMB = 22528 // 22 GB minus a 512 MiB safety margin.
	}
	if c.Logging.Level == "" {
		c.Logging.Level = "info"
	}
	if c.Logging.Format == "" {
		c.Logging.Format = "json"
	}
	if c.Logging.RotateMB == 0 {
		c.Logging.RotateMB = 100
	}
	if c.Logging.KeepFiles == 0 {
		c.Logging.KeepFiles = 5
	}
}

// Validate checks the invariants the rest of the system relies on: a bindable
// API address, at least one token and one model, every model pointing at a
// known backend with a base URL, and every alias and the default model
// resolving to a real entry.
func (c *Config) Validate() error {
	if c.Bind.APIAddr == "" {
		return fmt.Errorf("config: bind.api_addr is required")
	}
	if _, _, err := net.SplitHostPort(c.Bind.APIAddr); err != nil {
		return fmt.Errorf("config: bind.api_addr %q is not host:port: %w", c.Bind.APIAddr, err)
	}
	if _, _, err := net.SplitHostPort(c.Bind.AdminAddr); err != nil {
		return fmt.Errorf("config: bind.admin_addr %q is not host:port: %w", c.Bind.AdminAddr, err)
	}
	if len(c.Auth.Tokens) == 0 {
		return fmt.Errorf("config: auth.tokens needs at least one token")
	}
	for i, t := range c.Auth.Tokens {
		if strings.TrimSpace(t) == "" {
			return fmt.Errorf("config: auth.tokens[%d] is empty", i)
		}
	}
	if len(c.Models) == 0 {
		return fmt.Errorf("config: models needs at least one entry")
	}
	for name, m := range c.Models {
		if !validBackends[m.Backend] {
			return fmt.Errorf("config: model %q has unknown backend %q", name, m.Backend)
		}
		if m.BaseURL == "" {
			return fmt.Errorf("config: model %q is missing base_url", name)
		}
		if m.UpstreamModel == "" {
			return fmt.Errorf("config: model %q is missing upstream_model", name)
		}
		if m.VRAMMB < 0 {
			return fmt.Errorf("config: model %q has negative vram_mb", name)
		}
	}
	for alias, target := range c.Aliases {
		if _, ok := c.Aliases[target]; ok {
			return fmt.Errorf("config: alias %q points at another alias %q (one level only)", alias, target)
		}
		if _, ok := c.Models[target]; !ok {
			return fmt.Errorf("config: alias %q points at unknown model %q", alias, target)
		}
	}
	if c.DefaultModel == "" {
		return fmt.Errorf("config: default_model is required")
	}
	if !c.resolves(c.DefaultModel) {
		return fmt.Errorf("config: default_model %q is neither a model nor an alias", c.DefaultModel)
	}
	return nil
}

// resolves reports whether name is a known model id or a known alias.
func (c *Config) resolves(name string) bool {
	if _, ok := c.Models[name]; ok {
		return true
	}
	_, ok := c.Aliases[name]
	return ok
}
