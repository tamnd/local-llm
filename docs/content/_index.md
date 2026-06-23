---
title: "local-llm"
description: "local-llm is a reproducible single-box LLM stack for one machine. llmgw is a thin OpenAI-compatible gateway plus a VRAM steward that fronts Ollama, llama.cpp, ExLlamaV3/TabbyAPI, vLLM, and an in-process cgo engine behind one API and hot-swaps a single 24 GB GPU between models larger than fit together."
heroTitle: "One GPU, every open model, one API"
heroLead: "llmgw puts a single OpenAI-compatible endpoint in front of every inference backend on the box and hot-swaps the 24 GB GPU between models that are individually too big to share it. The card only ever holds the model you are using, plus a small embedding model kept resident for RAG. Your clients see one stable API, not five servers on five ports."
heroPrimaryURL: "/getting-started/quick-start/"
heroPrimaryText: "Get started"
---

A single RTX 4090 has 24 GB of VRAM, about 22 GB usable once the desktop and CUDA take their cut. That is enough to run one good model at a time, not every model you want at once. Running a separate server per model and remembering which port is which gets old fast. local-llm collapses that into one binary, `llmgw`, that speaks the OpenAI API and decides which backend and which model serves each request, swapping the GPU as it goes.

The whole thing lives on one workstation (GamingPC, RTX 4090, i9-13900K, 64 GB) reached only over a Tailscale private network. There is no public listener. The tailnet is the door.

## What it does

- **One stable OpenAI-compatible API.** `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`, with SSE streaming. Point any OpenAI client at it and change nothing else.
- **Hot-swaps one GPU between big models.** The 24 GB card holds only the model in use. Ask for a different model and `llmgw` evicts the current one and loads the next, so models that could never coexist all live behind the same endpoint.
- **Keeps an embedding model resident.** A small embedding model stays pinned in VRAM alongside the chat model, so RAG retrieval never pays a reload.
- **The tailnet is the only door in.** No public exposure. The data plane binds `0.0.0.0:8888`, the admin plane stays on loopback `8889`, and the only way to reach the box is over Tailscale (WireGuard, ACLs).
- **In-process zero-proxy engine.** An optional cgo build embeds llama.cpp so the gateway itself runs the decode loop with no subprocess and no HTTP hop, for maximum tokens per second on the dense tier.
- **Pure Go, one binary.** The default build is pure Go with a single dependency (yaml.v3) on Go 1.26. One file to ship, one file to run.

The local tier on this card is the ~30B class: 30-35B mixture-of-experts models with ~3-4B active params for the fast daily driver, and 24-32B dense models when you want peak quality. The 400B-1T frontier weights are out of scope to run whole.

## Where to go next

- New here? Start with the [introduction](/getting-started/introduction/), then the [quick start](/getting-started/quick-start/).
- Want to install it? See [installation](/getting-started/installation/).
- Tuning the stack? The [configuration reference](/reference/configuration/) covers the config schema and the in-process engine build.
- Need every flag? The [CLI reference](/reference/cli/) is the full surface.
