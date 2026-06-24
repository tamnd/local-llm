---
title: "Release notes"
description: "Per-version changelog for llmgw."
weight: 30
---

A release is cut by pushing a version tag. GoReleaser builds the binaries and
publishes platform archives, Linux packages (`.deb`, `.rpm`, `.apk`), a
container image, and Homebrew/Scoop entries. The version on the tag is stamped
into the binary, so `llmgw -version` reports what release it came from.

---

## v0.1.0

First numbered release. Contains all work from the initial build-out (PRs 1-9).

### What is in it

**Gateway and backends**

- `llmgw` HTTP gateway with an OpenAI-compatible `/v1/chat/completions`,
  `/v1/embeddings`, and `/v1/models` API over one or more backend models.
- Ollama backend: proxies to a running Ollama instance, with per-model routing
  and header translation.
- In-process (inproc) backend: runs llama.cpp inside the Go process via CGO,
  no proxy hop. Requires a CUDA-enabled `libllama.so` built by
  `scripts/build-libllama.sh`. Enabled with `-tags llama` at build time.
- Hot-swap manager: loads one model at a time within a VRAM budget; drains
  in-flight requests before unloading.
- Auth: bearer token list in config; separate admin token for management routes.

**Inproc engine patches for Ollama-format GGUFs**

The in-process engine links against llama.cpp b9780. Ollama's GGUF export
format differs from what b9780 expects for three new architectures. PR 9
ships a compatibility patch (`patches/llama-b9780-ollama-compat.patch`) that
`scripts/build-libllama.sh` applies after checkout:

- `gptoss` arch name (Ollama) vs `gpt-oss` (upstream): canonical name aligned,
  reverse alias added so old GGUFs still load.
- `attn_out` tensor name (Ollama gptoss) vs `attn_output` (upstream): new
  `LLM_TENSOR_ATTN_OUT_PROJ` type added.
- `ffn_norm.weight` (Ollama) used as pre-FFN norm instead of
  `post_attention_norm.weight`: fallback load into `attn_post_norm`.
- `attn_sinks` stored without `.weight` suffix in Ollama gptoss GGUFs.
- `rope.scaling.type` absent in Ollama gptoss GGUFs. llama.cpp defaults to
  linear/32 which collapses short-range positional encoding. The model uses
  NTK via a high base (150000), so patched to `rope_freq_scale_train = 1.0`.
- qwen35/qwen35moe: per-layer GQA for hybrid SSM/attention models (recurrent
  layers have zero KV heads; the global hparam cannot be used uniformly).
- qwen3.6:27b has 1307 GGUF tensors total; text-only loading reads 803, so
  `done_getting_tensors(partial=true)` is required.
- Partial array fills allowed with `required=false` (fixes
  `rope.dimension_sections` 3-element vs 4-element mismatch in qwen35moe).

**Build toolchain fixes**

- llama.cpp pin bumped from b6500 to b9780, which adds qwen35, qwen35moe, and
  gptoss architecture support.
- `CUDA_TOOLKIT_ROOT` pinned to `/usr/local/cuda-12` when multiple CUDA
  versions are installed. CUDA 13.x requires driver >= 590; GamingPC runs
  566.36 which supports 12.7 max.
- `-allow-unsupported-compiler` added to CUDA cmake flags. Ubuntu 26.04 LTS
  ships GCC 15 as the default compiler; CUDA 12.6 rejects it without this
  flag. The build then picks up GCC 12 via `-DCMAKE_CUDA_COMPILER=gcc-12`.

**Measured performance (RTX 4090, b9780, n_ctx=4096, Q8_0 KV)**

Dense 32B models (qwen2.5-coder:32b, deepseek-r1:32b):
- Ollama: ~26 tok/s (49% of bandwidth roofline)
- inproc: ~41 tok/s (78% of bandwidth roofline, +59%)

Small models (0.6B-8B): 94-97% of Ollama throughput.

New architectures via the b9780 patch:
- `qwen3.6:27b` (qwen35 hybrid SSM): 18.3 tok/s, 16040 MiB VRAM
- `qwen3.6:35b` (qwen35moe): 18.9 tok/s, 22236 MiB VRAM
- `gpt-oss:20b` MXFP4 (gptoss): 23.5 tok/s, 12992 MiB VRAM

The new-arch numbers are below Ollama (11-39% of Ollama decode speed). The
gap is from b9780's SSM and gptoss CUDA kernel paths being newly added and
not yet as optimized as the older qwen2moe/dense paths.

### Install

```sh
# macOS
brew install tamnd/tap/llmgw

# Windows
scoop bucket add tamnd https://github.com/tamnd/scoop-bucket
scoop install llmgw

# Linux (apt)
curl -fsSL https://tamnd.github.io/linux-repo/gpg.key | sudo gpg --dearmor -o /usr/share/keyrings/tamnd.gpg
echo "deb [signed-by=/usr/share/keyrings/tamnd.gpg] https://tamnd.github.io/linux-repo stable main" \
  | sudo tee /etc/apt/sources.list.d/tamnd.list
sudo apt update && sudo apt install llmgw

# Direct download
# https://github.com/tamnd/local-llm/releases/tag/v0.1.0
```
