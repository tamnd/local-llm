// Package router resolves the model name a client sends into the concrete
// model entry that names a backend, a base URL, and the upstream model id. It
// is pure lookup: alias expansion (one level), default-model substitution, and
// the stable model list for GET /v1/models (doc 08 section 4).
package router

import (
	"sort"

	"github.com/tamnd/local-llm/config"
)

// Router answers "given this model field, which backend entry serves it?". It is
// built once from the validated config and is read-only thereafter, so it needs
// no locking.
type Router struct {
	models       map[string]config.ModelEntry
	aliases      map[string]string
	defaultModel string
	ids          []string // sorted model ids for the models list
}

// New builds a Router from a validated config. Config.Validate has already
// guaranteed that every alias and the default resolve, so resolution here never
// has to defend against dangling references.
func New(cfg *config.Config) *Router {
	ids := make([]string, 0, len(cfg.Models))
	for id := range cfg.Models {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return &Router{
		models:       cfg.Models,
		aliases:      cfg.Aliases,
		defaultModel: cfg.DefaultModel,
		ids:          ids,
	}
}

// Resolved is the outcome of a successful lookup: the canonical model id and its
// backend entry.
type Resolved struct {
	ID    string
	Entry config.ModelEntry
}

// Resolve turns a client-supplied model name into a backend entry. An empty name
// falls back to the default model; an alias is expanded one level; the result is
// looked up in the model table. ok is false when the name matches nothing.
func (r *Router) Resolve(name string) (Resolved, bool) {
	if name == "" {
		name = r.defaultModel
	}
	canonical := name
	if target, ok := r.aliases[name]; ok {
		canonical = target
	}
	entry, ok := r.models[canonical]
	if !ok {
		// The name might itself be the default alias's target chain; try the
		// default once more in case default_model is an alias.
		if target, isAlias := r.aliases[canonical]; isAlias {
			canonical = target
			entry, ok = r.models[canonical]
		}
	}
	if !ok {
		return Resolved{}, false
	}
	return Resolved{ID: canonical, Entry: entry}, true
}

// IDs returns the sorted list of canonical model ids for the /v1/models
// response. Aliases are not listed; clients see the real model namespace.
func (r *Router) IDs() []string {
	out := make([]string, len(r.ids))
	copy(out, r.ids)
	return out
}

// Default returns the configured default model name (which may be an alias).
func (r *Router) Default() string { return r.defaultModel }
