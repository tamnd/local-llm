---
title: "Quick start"
description: "Write a minimal config, start llmgw, and get your first chat completion back over the OpenAI API."
weight: 30
---

This walks through a first run end to end: one config file with a single Ollama-backed model, starting the gateway, and calling it with `curl`. It assumes you have [installed llmgw](/getting-started/installation/) and have Ollama running on the box with a model pulled (`ollama pull llama3.1` for example).

## A minimal config

Save this as `llmgw.yaml`. It binds both planes, sets one auth token, and registers one model keyed `chat`, backed by Ollama.

```yaml
bind:
  api_addr: "0.0.0.0:8888"
  admin_addr: "127.0.0.1:8889"

auth:
  tokens:
    - "change-me-to-a-long-random-string"

models:
  chat:
    backend: ollama
    upstream_model: "llama3.1"
    base_url: "http://127.0.0.1:11434"
```

`models` is keyed by the name clients ask for, so here that key is `chat`. `backend` is one of `ollama`, `llama`, `tabby`, `vllm`, or `inproc`. `upstream_model` is the name the backend itself knows the model by. For the HTTP backends you give a `base_url`; for `inproc` you give `params.model_path` instead of a `base_url`.

## Start the gateway

```bash
llmgw -config llmgw.yaml
```

It comes up on `0.0.0.0:8888` for data and loopback `8889` for admin. Check it is alive:

```bash
curl http://localhost:8888/healthz
```

## First chat completion

Call the OpenAI endpoint with the bearer token from your config. Note the model name is the `models` key you set, `chat`, not the upstream model name.

```bash
curl http://localhost:8888/v1/chat/completions \
  -H "Authorization: Bearer change-me-to-a-long-random-string" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chat",
    "messages": [
      {"role": "user", "content": "Say hello in one short sentence."}
    ]
  }'
```

The first call may take a moment while llmgw loads the model onto the GPU. Subsequent calls are fast.

## List the models

```bash
curl http://localhost:8888/v1/models \
  -H "Authorization: Bearer change-me-to-a-long-random-string"
```

You get back the model names from your config (the `models` keys, plus any `aliases`), the same names you pass as `"model"`.

## Reaching it over the tailnet

The box is reached over Tailscale, so from any other machine on the tailnet swap `localhost` for the box's tailnet address:

```bash
curl http://100.71.238.128:8888/v1/chat/completions \
  -H "Authorization: Bearer change-me-to-a-long-random-string" \
  -H "Content-Type: application/json" \
  -d '{"model": "chat", "messages": [{"role": "user", "content": "Hello from the tailnet."}]}'
```

From here, add more models to the `models` block and llmgw will hot-swap the GPU between them on demand. See the [configuration reference](/reference/configuration/) for the full schema, multiple backends, and the in-process engine.
