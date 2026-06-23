---
title: "Hot-swapping models"
description: "One 24 GB card cannot hold every model at once, so llmgw loads on first use and swaps out when you ask for something else."
weight: 10
---

The RTX 4090 has 24 GB of VRAM, roughly 22 GB usable once you leave room for the
driver and KV cache. A 30B-class model fills most of that on its own, so you
cannot keep two large models resident at the same time. Instead of failing, the
gateway treats VRAM as a single seat: it loads a model the first time you ask for
it, and when you request a different one it unloads the old model and loads the
new one. From a client's point of view nothing changes, you just name the model
you want in the request body and the gateway makes sure it is the one in memory.

One thing does stay put across every swap: the small embedding model. The
gateway holds it resident next to whatever chat model is loaded, under a budget
called coexist. RAG traffic hits `/v1/embeddings` constantly, and if every
embedding call evicted the chat model the box would thrash. The coexist budget
reserves a slice of VRAM for the embedder so the two live side by side.

## See what is configured

The data plane on `8888` is what clients talk to. List the models the gateway
knows about:

```bash
curl -s http://100.71.238.128:8888/v1/models | jq .
```

You get back every configured model id and alias. Listing does not load anything,
it just reports what is available to request.

## Trigger a load

Send a chat request naming model A. If it is not already in VRAM, the gateway
loads it before answering, so the first call after a cold start or a swap is
slower while weights move onto the card:

```bash
curl -s http://100.71.238.128:8888/v1/chat/completions \
  -H "Authorization: Bearer $LLMGW_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-30b",
    "messages": [{"role": "user", "content": "ping"}]
  }'
```

## Trigger a swap

Now send a request naming model B. Because A and B cannot both fit, the gateway
unloads A and loads B. This second request pays the same load cost A did, while
requests for B after that run warm:

```bash
curl -s http://100.71.238.128:8888/v1/chat/completions \
  -H "Authorization: Bearer $LLMGW_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma3-27b",
    "messages": [{"role": "user", "content": "ping"}]
  }'
```

Through both of those swaps the embedding model never moved. An embedding call
in between the two chat requests would have been answered without disturbing
either chat model.

## Where the swap state lives

Load and unload are managed on the admin plane, which listens on loopback only at
`8889`. That separation is deliberate: clients on the tailnet drive the data
plane on `8888`, while the steward's view of what is resident and what is queued
stays on the box itself at `8889`. Health is exposed at `/healthz`:

```bash
curl -s http://127.0.0.1:8889/healthz
```

If you are debugging a slow first token, the admin plane is where you confirm
whether the model you asked for was already resident or had to be swapped in.
