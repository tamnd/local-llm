---
title: "CLI reference"
description: "The llmgw command-line flags and the HTTP routes it serves."
weight: 10
---

```bash
llmgw [flags]
```

`llmgw` is an OpenAI-compatible gateway and VRAM steward. It fronts the backends
named in its config (Ollama, llama-server, TabbyAPI, vLLM, and the in-process
engine), routes each request by model name, and swaps models in and out of VRAM
so one large model fits a 24 GB card. There are no subcommands: the binary reads
a config file and starts serving.

## Flags

`llmgw` defines two flags.

| Flag | Default | Meaning |
|------|---------|---------|
| `-config` | `configs/llmgw.yaml` | Path to the gateway config file. See the [configuration reference](/reference/configuration/). |
| `-version` | `false` | Print the version and exit. |

Start the gateway against an explicit config:

```bash
llmgw -config /etc/llmgw/llmgw.yaml
```

Print the version and exit:

```bash
llmgw -version
```

```text
llmgw v0.1.0
```

The version is stamped into the binary at release time. A build from source that
was not stamped reports `dev`. The same string feeds the `/healthz` response and
the `system_fingerprint` field on completions.

On startup `llmgw` loads and validates the config before either listener binds.
A missing or malformed field is a fatal error printed to stderr, never a surprise
at request time. The process shuts both listeners down gracefully on an interrupt
or a `SIGTERM`.

## Endpoints

`llmgw` binds two listeners that serve the same routes. The gateway enforces
which token may reach which route, so the split is about network reachability,
not a second copy of the API.

### Data plane

The data plane binds `0.0.0.0:8888` by default (`bind.api_addr`). The address is
open because the only route to the port is the tailnet; the Tailscale network is
the security boundary. Every data-plane request needs a bearer token from
`auth.tokens`.

| Method | Route | Purpose |
|--------|-------|---------|
| `POST` | `/v1/chat/completions` | Chat completion |
| `POST` | `/v1/completions` | Text completion |
| `POST` | `/v1/embeddings` | Embeddings |
| `GET` | `/v1/models` | List the models the gateway serves |

The chat and completion routes stream when the request body sets `"stream":
true`. Streaming responses are sent as server-sent events, the same
`data: {...}` chunk framing an OpenAI client expects, terminated by `data:
[DONE]`. With `"stream": false` (or the field omitted) the route returns one JSON
body.

```bash
curl http://gamingpc:8888/v1/chat/completions \
  -H "Authorization: Bearer $LLMGW_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chat",
    "stream": true,
    "messages": [{"role": "user", "content": "hello"}]
  }'
```

The `model` field takes a model id or an alias from the config. A request that
omits `model` is answered by `default_model`.

### Admin plane

The admin plane binds `127.0.0.1:8889` by default (`bind.admin_addr`). It stays
on loopback so the force-load and force-unload controls are unreachable from the
network. Admin routes need the `auth.admin_token`; if that token is empty the
admin plane is disabled.

| Method | Route | Purpose |
|--------|-------|---------|
| `GET` | `/admin/status` | Current slot, queue depth, and VRAM accounting |
| `POST` | `/admin/load` | Force a model into VRAM |
| `POST` | `/admin/unload` | Evict a model from VRAM |

### Health

`GET /healthz` reports liveness and the running version. It needs no token and is
served on both planes.

```bash
curl http://127.0.0.1:8889/healthz
```
