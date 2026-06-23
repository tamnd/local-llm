---
title: "Configuration"
description: "The llmgw YAML config schema: bind, auth, models, aliases, and the in-process engine."
weight: 20
---

`llmgw` is configured by a single YAML file, passed with `-config` (default
`configs/llmgw.yaml`). The file is decoded into the `config.Config` struct,
defaults are applied, and the whole thing is validated once at startup. An
unknown key is rejected, so a typo fails fast rather than being silently ignored.
A minimal config that names an API bind address, one token, and one model is
enough to start.

The blocks below are the top-level keys: `bind`, `auth`, `models`, `aliases`,
plus `default_model`, `manager`, and `logging`.

## bind

The two listening sockets. The data plane is reachable over the tailnet; the
admin plane stays on loopback.

| Field | Default | Meaning |
|-------|---------|---------|
| `api_addr` | required | `host:port` for the data plane. `0.0.0.0:8888` is fine because the tailnet is the only route to the port. Set it to the Tailscale IP to be strict. |
| `admin_addr` | `127.0.0.1:8889` | `host:port` for the admin plane. Keep it on loopback so force-load and force-unload cannot be reached from the network. |

Both addresses must parse as `host:port`. `api_addr` is required; `admin_addr`
defaults when left out.

## auth

The bearer-token policy.

| Field | Default | Meaning |
|-------|---------|---------|
| `tokens` | required | List of tokens that authenticate the data plane. At least one is required, and no entry may be blank. |
| `admin_token` | empty | Single token guarding the admin routes. Leave it empty to disable the admin plane entirely. |

Generate a token with `openssl rand -hex 24`.

## default_model

The model id or alias that answers a request which omits the `model` field. It is
required and must resolve to a real model or alias.

## aliases

A map of friendly name to model id. A client may send the alias as the `model`
field. An alias must point at a real model entry, and it may point only at a
model, not at another alias (one level of indirection).

## models

A map of model id to a model entry. Clients select an entry by its id or by an
alias. Each entry names the backend that owns the model, where to reach it, and
what the backend calls it.

| Field | Required | Meaning |
|-------|----------|---------|
| `backend` | yes | One of `ollama`, `llama`, `tabby`, `vllm`, `inproc`. |
| `base_url` | HTTP backends | Where the backend listens, for example `http://127.0.0.1:11434`. Required for every backend except `inproc`. |
| `upstream_model` | yes | The name the backend uses for the model, forwarded on each request. |
| `vram_mb` | no | Measured VRAM residency in MiB. The manager uses it for the coexistence check. Must not be negative. |
| `coexist` | no | When `true`, the model may stay resident alongside the main slot instead of being evicted on a swap. Useful for a small embedding model. |
| `params` | no | A free-form map passed to the backend adapter. The keys a backend reads depend on the backend (see below). |

Which fields a backend requires:

- The HTTP backends (`ollama`, `llama`, `tabby`, `vllm`) need `base_url`. The
  gateway forwards requests to that URL.
- The in-process backend (`inproc`) has no network endpoint. It loads a GGUF from
  disk, so instead of `base_url` it needs `params.model_path`. A config that
  names `backend: inproc` without `params.model_path` fails validation.

### params per backend

The `params` map is forwarded as-is to the backend adapter, so the keys that
matter depend on the backend. The ones used in the shipped config and catalog:

| Backend | Param | Meaning |
|---------|-------|---------|
| `ollama` | `keep_alive` | How long Ollama keeps the model resident after the last request, for example `30m`. |
| `tabby` | `admin_key` | TabbyAPI admin key. |
| `tabby` | `max_seq_len` | Maximum sequence length TabbyAPI loads the model with. |
| `tabby` | `cache_mode` | KV-cache quantization mode, for example `Q8`. |
| `inproc` | `model_path` | Path to the GGUF the in-process engine loads. Required for `inproc`. |
| `inproc` | `draft_path` | Optional path to a vocab-matched draft GGUF for speculative decoding. |
| `inproc` | `n_ctx` | Context length. |
| `inproc` | `n_gpu_layers` | Layers offloaded to the GPU (`99` offloads all). |
| `inproc` | `flash_attn` | Enable flash attention. |
| `inproc` | `cache_type_k` | KV-cache type for keys, for example `q8_0`. |
| `inproc` | `cache_type_v` | KV-cache type for values, for example `q8_0`. |
| `inproc` | `draft_max` | Maximum number of draft tokens proposed per step. |
| `inproc` | `temperature` | Sampler temperature. Speculative decoding engages only at `0.0` (greedy). |

## manager

The swap and queue policy. Every field has a default, so the whole block is
optional.

| Field | Default | Meaning |
|-------|---------|---------|
| `hot_swap` | `false` | Unload the outgoing model before loading the next one. |
| `vram_budget_mb` | `22528` | VRAM ceiling the coexistence check measures against (22 GB minus a 512 MiB margin on the 24 GB card). |
| `queue_max` | `32` | Requests allowed to wait behind an in-flight swap. |
| `drain_timeout_s` | `120` | How long to wait for in-flight requests before a swap. |
| `load_timeout_s` | `120` | How long a load may take. Process-driven backends like vLLM can take minutes. |
| `unload_timeout_s` | `30` | How long an unload may take. |

## logging

The structured request log.

| Field | Default | Meaning |
|-------|---------|---------|
| `level` | `info` | Log level. |
| `format` | `json` | Log format. |
| `file` | empty | Log file path. Empty logs to stdout. |
| `rotate_mb` | `100` | Rotate the log file once it passes this size in MiB. |
| `keep_files` | `5` | How many rotated files to keep. |

## Example

A complete config, adapted from `configs/llmgw.yaml`:

```yaml
bind:
  # The data plane is open because the tailnet is the only route to the port.
  api_addr: "0.0.0.0:8888"
  # The admin plane stays on loopback.
  admin_addr: "127.0.0.1:8889"

auth:
  # Generate with: openssl rand -hex 24
  tokens:
    - "REPLACE_ME_data_plane_token"
  # Empty admin_token disables the admin plane.
  admin_token: "REPLACE_ME_admin_token"

# Answers requests that omit the model field. May be an alias.
default_model: chat

aliases:
  chat: qwen3-30b-a3b
  coder: qwen3-coder-30b-a3b
  reasoning: qwen3-32b-exl2
  embed: nomic-embed-text

models:
  # General chat and tool use, MoE, served by Ollama.
  qwen3-30b-a3b:
    backend: ollama
    base_url: "http://127.0.0.1:11434"
    upstream_model: "qwen3:30b-a3b"
    vram_mb: 19000
    params:
      keep_alive: "30m"

  # Dense 32B at 4bpw EXL2 through TabbyAPI: higher quality, slower.
  qwen3-32b-exl2:
    backend: tabby
    base_url: "http://127.0.0.1:5000"
    upstream_model: "Qwen3-32B-exl2-4.0bpw"
    vram_mb: 20500
    params:
      admin_key: "REPLACE_ME_tabby_admin_key"
      max_seq_len: 32768
      cache_mode: "Q8"

  # Embeddings, small enough to stay resident alongside the main slot.
  nomic-embed-text:
    backend: ollama
    base_url: "http://127.0.0.1:11434"
    upstream_model: "nomic-embed-text"
    vram_mb: 350
    coexist: true
    params:
      keep_alive: "60m"

  # The in-process zero-proxy path. No base_url; it loads the GGUF directly.
  qwen3-coder-inproc:
    backend: inproc
    upstream_model: qwen3-coder-30b
    vram_mb: 21000
    params:
      model_path: "/models/gguf/Qwen3-Coder-30B-UD-Q4_K_XL.gguf"
      draft_path: "/models/gguf/Qwen3-1.7B-Q4_K_M.gguf"  # must share the target vocab
      n_ctx: 32768
      n_gpu_layers: 99
      flash_attn: true
      cache_type_k: "q8_0"
      cache_type_v: "q8_0"
      draft_max: 6
      temperature: 0.0  # speculative decoding engages only in greedy mode

manager:
  hot_swap: true
  vram_budget_mb: 22528
  queue_max: 32
  drain_timeout_s: 120
  load_timeout_s: 300
  unload_timeout_s: 30

logging:
  level: info
  format: json
  file: "/var/log/llmgw/requests.log"  # empty logs to stdout
  rotate_mb: 100
  keep_files: 5
```

## The in-process engine

The `inproc` backend is the zero-proxy, high-throughput path. Instead of
forwarding over HTTP to a separate process, it loads the GGUF straight into the
gateway process through a cgo binding of llama.cpp, so there is no Ollama
subprocess and no HTTP hop, and the gateway controls the sampler, the KV-cache
types, and speculative decoding directly.

It needs a cgo build of the binary. Compile a CUDA-linked `libllama` with
`scripts/build-libllama.sh` (the `make libllama` target), then build the gateway
against it with `make build-llama`. A default build, without those tags, reports
the in-process engine as unavailable.

A model entry for `inproc` points `params.model_path` at a GGUF on disk. It takes
an optional `draft_path` pointing at a vocab-matched draft model: with a draft
present and `temperature: 0.0`, the engine runs greedy speculative decoding,
which only pays off in greedy mode. The recommended quant is an Unsloth
UD-Q4_K_XL dynamic quant, which gives the best quality per byte at the same
decode speed. The catalog under `catalog/models.manifest.yaml` lists the GGUF and
draft files each in-process entry is provisioned against.
