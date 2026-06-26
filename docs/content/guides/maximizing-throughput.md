---
title: "Maximizing throughput"
description: "The zero-proxy path: embed llama.cpp in the gateway and run speculative decoding to push the dense tier toward its bandwidth ceiling."
weight: 30
---

For everyday work the MoE model is the daily driver and it is already fast. A
30-35B MoE with only 3-4B active parameters reads a small slice of weights per
token, so it runs hundreds of tokens per second and you rarely think about
throughput. The dense tier is the one worth tuning. A 24-32B dense model reads
its full weight set on every decode step, which makes decode memory-bandwidth
bound: tokens per second is roughly the card's memory bandwidth divided by the
bytes read per token. That is a hard roofline set by the hardware, and the only
levers you have are reading fewer bytes per token or doing useful work while the
bandwidth is busy.

## Why batch=1 Ollama leaves throughput on the table

Ollama is a fine Windows-native fallback and it is easy to run, but at batch=1 on
the dense tier it sits well under the bandwidth roofline, in practice around half
of it, and it does not do speculative decoding. So you pay the full per-token
read cost and only get one token out of each pass. Two things are being left
unused: the gap up to the roofline, and the chance to emit more than one verified
token per target pass.

## The zero-proxy answer: the inproc backend

The `inproc` backend embeds llama.cpp directly in the gateway process through
cgo. There is no subprocess to manage and no HTTP hop between the gateway and the
engine, so a request goes straight into the decode loop instead of crossing a
local socket. On the dense tier, where every microsecond of overhead competes
with bandwidth-bound work, removing the proxy hop matters.

On top of that, `inproc` does greedy speculative decoding at temperature 0. A
small vocab-matched draft model, for example a 1.7B draft in front of a 27-30B
target, proposes several tokens, and the target verifies them in one pass. At
temperature 0 the verification is exact: the output is identical to what the
target would have produced on its own, you just get there in fewer target passes
when the draft guesses right. The draft must share the target's vocabulary for
this to work.

Two model-shape rules follow from that:

- Dense targets benefit from a separate small draft. Match the vocabulary and
  keep the draft small enough that proposing is cheap.
- MoE models should not use a separate draft. Use their built-in MTP if they have
  it, or nothing. Bolting an external draft onto a MoE that is already fast just
  adds overhead.

Unsloth dynamic quants are the other lever on bytes per token. UD-Q4_K_XL gives
more quality per byte than a flat quant at the same decode speed, so you read the
same number of bytes but the model behaves like a higher-quality one. That is
free quality on a bandwidth-bound path.

## Enabling inproc

Download `llmgw-cuda` from the [GitHub release page](https://github.com/tamnd/local-llm/releases) and run it directly — it statically embeds llama.cpp and cuBLAS so there is nothing else to install on the box.

If you need to build from source:

```bash
scripts/build-libllama.sh --static
make build-llama-static
# produces bin/llmgw-cuda
```

Then add a model entry that uses the backend. Unlike the HTTP backends there is
no `base_url`, you point at the weight files on disk with `params.model_path`,
and you can add a draft path to turn on speculative decoding:

```yaml
models:
  qwen3-coder-inproc:
    backend: inproc
    upstream_model: qwen3-coder-30b
    params:
      model_path: "/models/gguf/Qwen3-Coder-30B-UD-Q4_K_XL.gguf"
      draft_path: "/models/gguf/Qwen3-1.7B-Q4_K_M.gguf"   # must share the target vocab
      temperature: 0.0   # speculative decoding engages only in greedy mode
```

Drop the `draft_path` line for a MoE model, or any model where you do not want a
separate draft.

## Measured numbers (RTX 4090, b9811, n_ctx=4096, Q8_0 KV)

Dense 32B models (qwen2.5-coder:32b, deepseek-r1:32b):

| Backend | tok/s | vs roofline |
|---------|-------|-------------|
| Ollama | ~26 | 49% |
| inproc | ~41 | 78% |

That 59% gain comes from removing the Ollama subprocess, the local HTTP hop, and the batch=1 constraint, not from any change to the model weights.

For MoE models: the b9811 llama.cpp pin adds batched MoE dispatch and fused SSM kernels. Earlier builds (b9780) treated MoE decode the same as dense and saturated around 18-23 tok/s. b9811 closes most of that gap for MoE-shaped workloads. For MoE and SSM-hybrid models on this card, the `vllm` backend with FlashInfer is the current recommended path for maximum throughput, and it avoids the b9780-era kernel maturity gap entirely.

## vLLM as the MoE alternative

For MoE-heavy models (Qwen3.5-35B, gpt-oss-20b) the `vllm` backend with FlashInfer chunked prefill often beats inproc, because FlashInfer has mature MoE dispatch kernels that predate llama.cpp's:

```yaml
models:
  qwen3-14b-fp8:
    backend: vllm
    base_url: "http://127.0.0.1:8100"
    upstream_model: "/models/hf/Qwen3-14B-FP8"
    vram_mb: 20000
    params:
      gpu_memory_utilization: "0.85"
      max_model_len: "16384"
      enable_chunked_prefill: "true"
```

`scripts/08-vllm.sh` sets up the vLLM install and the systemd units. Use inproc for dense models, vLLM for MoE or quantized models where it has the better kernel.
