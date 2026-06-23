---
title: "Introduction"
description: "Why one gateway in front of many backends beats one server per model, and how llmgw hot-swaps a single 24 GB GPU."
weight: 10
---

A 24 GB GPU is a fixed budget. About 22 GB is usable once the desktop and the CUDA runtime take their share, and that is enough for one good model at a time. The models worth running are individually large: a 30-35B mixture-of-experts model, or a 24-32B dense model, will each fill most of the card on its own. You cannot hold them all at once, so you have to choose which one is loaded.

The usual workaround is to run a separate inference server per model: Ollama here, llama-server there, TabbyAPI or vLLM somewhere else, each on its own port. Now every client has to know which model lives on which port, and you are manually loading and unloading models to keep the GPU from overflowing. The endpoint you call changes depending on what you want, which is exactly backwards.

llmgw fixes the shape. It is one OpenAI-compatible API in front of all of those backends. A request names a model; llmgw routes it to whichever backend holds that model, and if that model is not the one currently resident it evicts the loaded model and loads the one you asked for. The GPU only ever holds the model in use, plus a small embedding model kept pinned so RAG retrieval never triggers a reload. From the client side there is one base URL and one stable set of endpoints. The swapping is the gateway's problem, not yours.

The box itself is private. local-llm runs on one workstation reached only over a Tailscale network, so there is no public listener to harden. The data plane binds `0.0.0.0:8888` and the admin plane stays on loopback. The tailnet, WireGuard plus ACLs, is the security boundary.

Next: [install llmgw](/getting-started/installation/).
