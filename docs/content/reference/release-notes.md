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

## v0.2.0

PRs 11-16. Adds the vLLM backend, a quality benchmark suite, bumps llama.cpp
to b9811, and ships a prebuilt static CUDA binary.

### vLLM backend

`backend: vllm` routes requests to a running vLLM process. vLLM uses FlashInfer
fused attention kernels and chunked prefill, which makes it the preferred path
for FP8 and MXFP4 quantized models where llama.cpp does not yet have optimized
kernel paths. The gateway manages load and unload through vLLM's health endpoint
and kills or restarts the process on swap.

Two model entries ship in the default config:

- `qwen3-14b-fp8`: Qwen3-14B in FP8 precision, 15.21 GiB, max 16384 tokens.
  Port 8100. GPU utilization 0.85.
- `gpt-oss-20b-vllm`: Microsoft gpt-oss-20b in native MXFP4 precision, 12.8 GiB,
  max 8192 tokens, FP8 KV cache. Port 8101. GPU utilization 0.92.

`scripts/08-vllm.sh` installs vLLM (pinned to 0.23.0) and writes systemd units
for both models. Enable and start each unit to bring it up. The gateway's
`load_timeout_s` defaults to 300 for vLLM because CUDA graph compilation takes
one to three minutes on first load.

vLLM params the gateway reads from the model entry's `params` block:

| Param | Meaning |
|-------|---------|
| `gpu_memory_utilization` | Fraction of GPU memory vLLM may use (e.g. `"0.85"`). |
| `max_model_len` | Maximum sequence length (tokens). |
| `kv_cache_dtype` | KV-cache quantization, e.g. `"fp8"`. |
| `enable_chunked_prefill` | `"true"` enables chunked prefill. |

### Quality benchmark suite

`scripts/bench-quality.py` runs MMLU (57 subjects), GSM8K, and HumanEval
against the gateway using the standard zero-shot prompts and compares results
across models and backends. `scripts/run-bench-quality.sh` drives it against all
models in a single run and writes a JSON summary.

The suite requires only a running gateway on port 8888. It can compare an
Ollama-backed model against a vLLM-backed one serving the same weights to check
that the backends agree. Run with `--model chat` to benchmark the default model,
or name multiple models to run a head-to-head comparison.

### llama.cpp bumped to b9811

The llama.cpp pin moves from b9780 to `9df06805` (b9811). This commit adds
batched MoE dispatch and fused SSM kernels, which give a large throughput
improvement for the Qwen3.5 hybrid SSM models and MoE models where b9780's
kernel paths were newly added and not yet optimized.

The Ollama-compat patch (`patches/llama-b9780-ollama-compat.patch`) applied
cleanly at b9811 with one-line fuzz.

### Static CUDA single-binary (llmgw-cuda)

`make build-llama-static` links cuBLAS, cudart, and llama.cpp into the binary
itself. The output (`bin/llmgw-cuda`) is a single 698 MB ELF that dynamically
links only `libcuda.so.1` (the NVIDIA driver, always present) and needs no
`libllama.so` on `LD_LIBRARY_PATH`.

`llmgw-cuda` is attached to each GitHub release as a standalone download. Drop
it on any Linux box with an NVIDIA driver ≥ 525 and run it directly. It
supports the same config format and all the same backends as the default binary,
plus the `inproc` backend which runs llama.cpp in-process.

The static build uses CUDA 12 by default. CUDA 13 is supported on machines
where the driver is ≥ 590 (override with `CUDA_ROOT=/usr/local/cuda-13`).

### Install

```sh
# macOS (pure-Go binary, no inproc engine)
brew install tamnd/tap/local-llm

# Windows
scoop bucket add tamnd https://github.com/tamnd/scoop-bucket
scoop install local-llm

# Linux (apt)
curl -fsSL https://tamnd.github.io/linux-repo/gpg.key | sudo gpg --dearmor -o /usr/share/keyrings/tamnd.gpg
echo "deb [signed-by=/usr/share/keyrings/tamnd.gpg] https://tamnd.github.io/linux-repo stable main" \
  | sudo tee /etc/apt/sources.list.d/tamnd.list
sudo apt update && sudo apt install local-llm

# Static CUDA binary with inproc engine (Linux x86_64 + NVIDIA GPU)
# Download llmgw-cuda from the release page
# https://github.com/tamnd/local-llm/releases/tag/v0.2.0
```

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
