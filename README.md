# local-llm

A reproducible, single-box local-LLM stack for one specific machine: a personal
workstation called **GamingPC**, built around an RTX 4090, reachable only over a private
Tailscale network. This repo turns that one box into a private, always-on,
OpenAI-compatible inference endpoint you can call from a laptop, a phone, an IDE, or a CLI
anywhere on the tailnet.

It is not a cloud service, not a multi-tenant platform, and not a training rig. It is the
catalog, the runtime configs, the provisioning scripts, the unified gateway, the benchmark
harness, and the runbooks that make one consumer GPU serve the modern open-weight stack
behind one stable API.

## The thesis in one line

**One consumer GPU, one private network, the full modern open-weight stack, served behind
one OpenAI-compatible API.**

The local story on a single 24 GB RTX 4090 is the ~30B-class tier: 30-35B mixture-of-experts
models with ~3-4B active parameters (fast, daily-driver and agentic-coding quality), and
24-32B dense models (slower per token, peak quality, multimodal). The 400B-1T frontier
open weights of 2026 (GLM-5.2, Kimi K2.7, DeepSeek-V4, Mistral Large 3, Qwen3.5-397B) need
multi-GPU datacenter VRAM and are out of scope to run whole; they are the ceiling this box
is measured against, not what it serves.

## The box

RTX 4090 24 GB (Ada, compute 8.9, ~1008 GB/s, native FP8, no native FP4); i9-13900K
(8P+16E, 32T); 64 GB DDR5-5600; 1 TB Kingston KC3000 NVMe; Windows 11 Pro. Reached over
Tailscale at `100.71.238.128`. Usable VRAM budget is about 22 GB after the desktop and
CUDA overhead. Full hardware detail and the memory-bandwidth roofline live in the spec.

## Layout

| Path | What it holds |
|------|---------------|
| `catalog/` | The curated model manifest: model, params, quant, VRAM, tok/s, license, install tag |
| `configs/` | Per-runtime config (Ollama env, llama.cpp flags, ExLlamaV3/TabbyAPI, vLLM, SGLang) |
| `gateway/` | The unified OpenAI-compatible gateway: model routing, hot-swap, the API surface |
| `scripts/` | Idempotent provisioning for the Windows host: drivers, runtimes, services, downloads |
| `bench/` | The benchmark harness and recorded results (TTFT, decode tok/s, concurrent throughput) |
| `docs/` | Operator-facing runbooks distilled from the spec |

## Design of record

The full design lives in the specification at `~/notes/Spec/2065/`. It is concrete: it
names the VRAM arithmetic down to bytes per weight and KV-cache bytes per token, the model
catalog down to install tags and measured token rates, every runtime's capability matrix,
the gateway's API surface, and the Tailscale security model. A second engineer should be
able to stand the box up from those documents alone. When this repo and the spec disagree,
the spec is the design and the repo is the bug until a document is updated to match a
deliberate change.

## Status

Specification phase. The spec is complete; the repo implementation follows the roadmap in
the spec's `14-roadmap.md`.

## License

MIT. See [LICENSE](LICENSE).
