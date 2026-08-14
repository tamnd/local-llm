# bench

Throughput and real-task measurement for the models the gateway serves.
The methodology and the results ledger live in spec 2065 doc 19; this is the operator's quick reference.

## Tools

- `cmd/llmbench` (Go): cross-backend throughput and context sweep. Drives the gateway's OpenAI endpoint, so one run covers ollama, vLLM, and ExLlamaV3 behind the same API. Records decode tok/s, prefill tok/s, TTFT, and (out of band) VRAM to `bench/results/<date>.jsonl`.
- `scripts/solve-swebench.sh`: real-task solve rate. Points the tomo-labs swebench-live harness at the gateway once per backend model and folds each model's solve rate, tokens, and rounds to `bench/results/solve-<date>.jsonl`.
- `ollama-bench.ps1`: the original ollama-only decode bench (server-side timing). Superseded by `llmbench` for cross-backend work; kept for the native-Windows ollama path.
- `thresholds.yaml`: the per-model performance gates (min decode tok/s, max TTFT, max swap seconds).

## Throughput

From the Mac, against the box gateway:

```
LLMBENCH_TOKEN=<token> go run ./cmd/llmbench \
  -models qwen3-32b,qwen3-32b-awq,qwen3-32b-exl3,qwen3-30b-a3b,qwen3-30b-a3b-awq,qwen3-30b-a3b-exl3 \
  -vram-cmd 'ssh box "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"'
```

Flags: `-base-url` (default the box tailnet gateway), `-config` (labels each row's runtime and quant from the gateway config), `-contexts`, `-gen`, `-reps`, `-out`. `-dry-run` prints the plan without calling the endpoint. `-arm` and `-runtime-ver` label a row when the same model is measured under more than one server configuration. Vision cells come from `-images` and `-vision-prompt` (see below).

### Prompt caching and the prefill column

Every rep is prefixed with a per-rep salt (`[rep 2] ...`) so it is a distinct
prompt. This is not cosmetic. llama-server caches by longest common prefix, so
identical reps make reps 2 and 3 skip prefill almost entirely, and the reported
prefill tok/s and TTFT then describe a cache hit rather than the model. At 32K
on this box the difference is 8.6 s of real prefill against 0.7 s cached, which
is why the 33.8k tok/s figure this file used to carry for Muse Glimmer was about
10x the truth. Decode tok/s was never affected: it is measured over generated
tokens after the first.

`-reuse-prompt` restores the old behaviour, which is worth having for two
reasons: it reproduces pre-2026-08-12 rows, and a cached-prefill number is the
right one to quote when the question is what a warm agent loop feels like rather
than what the GPU can do. Rows measured that way are labelled in the `arm` field.

## Muse Glimmer 30B (2026-08-10)

First numbers for Meta's Muse Glimmer 30B, Unsloth `UD-Q4_K_XL` on llama-server
b10355 (WSL2, CUDA 13.1, sm_89). Rows are in `results/20260810.jsonl`; the
reference arm is the box's current `chat` default on the same sweep and the same
day. `qwen3-32b` is carried from `results/20260719.jsonl` as the dense yardstick.

| model | ctx 512 | 2048 | 8192 | 32768 | prefill @32K | VRAM |
|---|---|---|---|---|---|---|
| muse-glimmer-30b | 50.8 | 50.5 | 49.2 | 46.6 | 3.3k tok/s | 15.1 GB |
| qwen3-30b-a3b (MoE, ollama) | 212.5 | 205.9 | 183.5 | 121.5 | 38.6k tok/s | 18.9 GB |
| qwen3-32b (dense, ollama) | 41.9 | 41.2 | 38.7 | - | 9.9k tok/s | 21.2 GB |

The prefill column for `muse-glimmer-30b` is corrected from the 33.8k tok/s
originally recorded here; that was a cache hit, not a measurement. The
replacement is the matching text-only arm in `results/20260812.jsonl`, which
runs the same flags on b10380 (see the note above). The two
ollama rows are unaudited and almost certainly carry the same inflation, so
treat this whole column as comparable only within a row until they are re-run.
The decode columns stand as measured.

Decode tok/s at batch 1, gen 256, 3 reps after a warm-up. Read it as: Muse
Glimmer is a ~28B *dense* model (llama-bench reports 27.85 B params, not a
sparse MoE), so it lands where dense models land, about 21% ahead of `qwen3-32b`
and about 4x behind the 3B-active MoE. It holds 46.6 tok/s at 32K, a 8.3% falloff
from 512 where the MoE loses 43%, and it leaves ~7 GB of the card free.

Raw runtime numbers without the gateway hop, for comparison with upstream:

```
llama-bench -m Muse-Glimmer-30B-UD-Q4_K_XL.gguf -ngl 999 -fa 1 -ctk q8_0 -ctv q8_0
  pp512   3479.07 ± 452.32 tok/s
  pp8192  3615.49 ±   4.23 tok/s
  tg128     51.67 ±   0.05 tok/s
```

Two things this box cannot currently do with the model, both upstream gaps
rather than config problems:

- **DFlash speculative decoding is a no-op.** llama.cpp loads the draft head and
  then fails it with `dflash requires ctx_other to be set`; measured decode is
  inside noise of the baseline and the head never becomes resident. See the
  commented-out `muse-glimmer-30b-dflash` entry in `configs/llmgw.yaml`.
- **Ollama cannot serve it on CUDA.** As of 0.32.7 the registry 412s every
  `muse-glimmer` tag on this host: support ships in the MLX engine on Apple
  Silicon only. Hence the llama-server path. Still true on 0.32.9.

## Muse Glimmer: the Unsloth recipe (2026-08-12)

The 08-10 run served the weights but not the model: it was missing `--mmproj`,
so the vision half of a multimodal model was dead, and it used llama-server's
default sampler rather than the model card's. This run applies Unsloth's
documented recipe (BF16 projector, temp 1.0 / top-p 0.95 / top-k 64) on
llama.cpp b10380, and separates that change from the runtime bump by measuring
the old flags on the new build too. Rows are in `results/20260812.jsonl`,
labelled by `arm`.

| arm | 512 | 2048 | 8192 | 32768 | prefill @32K | VRAM |
|---|---|---|---|---|---|---|
| b10380, pr22 flags (text only) | 50.8 | 50.7 | 48.9 | 46.7 | 3.3k | 15.1 GB |
| b10380 + projector + recipe sampling | 51.2 | 50.8 | 48.8 | 46.6 | 3.4k | 19.0 GB |

Three findings, in descending order of how much they matter:

- **Vision costs 3.9 GB and nothing else.** Decode is flat across the two arms
  at every context (worst case 48.9 → 48.8 tok/s, inside the run-to-run spread),
  so the projector is pure resident cost, not a throughput tax. It takes the
  model from 15.1 GB to 19.0 GB, and to ~21 GB after the first image allocates
  the mtmd encode buffers, which is what `vram_mb: 21000` in the config budgets
  for. That still leaves headroom on a 24 GB card, but it is no longer the
  roomiest 30B-class entry on the box.
- **b10355 → b10380 is a no-op for throughput.** Same weights, same flags,
  decode within 0.3 tok/s at every context. The build is worth taking for the
  multimodal CLI target and the accumulated fixes, not for speed.
- **DFlash is still a no-op.** Re-tested on b10380: 50.9 / 47.0 tok/s at 512 and
  32K against 51.2 / 46.6 without the draft head, the same `dflash requires
  ctx_other to be set` at load, and the same 19469 MB resident. Unchanged from
  b10355; the config entry stays commented out.

### Vision

Measured against generated fixtures (`bench/vision-fixtures.py`, deterministic,
committed under `fixtures/`), one image plus a short instruction per request:

| image | image tokens | prefill tok/s | TTFT | decode tok/s |
|---|---|---|---|---|
| 512x512 | 363 | 760 | 570 ms | 49.3 |
| 1024x1024 | 1371 | 1237 | 1.2 s | 51.4 |
| 2048x2048 | 4098 | 1134 | 3.7 s | 50.5 |

Read it as: an image is expensive to *ingest* and free to *reason over*. Decode
is identical to the text-only arm, so once the image is encoded the model runs
at its normal speed. But image tokens prefill at roughly 1.1-1.2k tok/s against
2.9-3.4k for text, about 2.6x slower per token, because the projector's encode
pass is on the critical path and is not batched the way text prefill is. A
2048x2048 image is 4098 tokens, so a single one costs about an eighth of the
32K window and 3.7 s before the first token appears. Downscale on the client
side unless the detail is load-bearing; 1024x1024 is the sweet spot here.

The `image_toks` column is derived, not estimated: each vision cell runs a
structurally identical text-only control through the same endpoint and
subtracts its `prompt_tokens`. If the difference is zero the cell fails loudly
rather than recording a row, which is the check that would have caught the
08-10 configuration -- a server without `--mmproj` answers image requests
happily, from the text alone, and looks healthy while doing it. The fixtures
each render a word the model is asked to read back, so a passing run also
demonstrates the projector is actually wired up and not merely allocating.

```
LLMBENCH_TOKEN=<token> go run ./cmd/llmbench -models muse-glimmer-30b \
  -images bench/fixtures/vision-512.png,bench/fixtures/vision-1024.png \
  -arm 'b10380+mmproj (unsloth recipe)' -runtime-ver b10380
```

## Qwen3.8 27B (2026-08-14)

Two checkpoints of the same model, taken as far as this card allows: Unsloth's
`UD-Q4_K_XL` GGUF on llama-server, and Qwen's native FP8 on vLLM. Only the first
one serves. Rows are in `results/20260814.jsonl`.

Qwen3.8 is a hybrid-attention VLM: 64 layers, of which 16 are full attention and
48 are Gated DeltaNet linear attention (`full_attention_interval: 4`). That
matters for every number below, because only the 16 full-attention layers hold a
growing KV cache; the 48 GDN layers hold a fixed-size recurrent state per
sequence. Native context is 262144 and the model is natively multimodal.

| arm | 512 | 2048 | 8192 | 32768 | prefill @32K | VRAM |
|---|---|---|---|---|---|---|
| gguf-ud-q4_k_xl, b10431 + mmproj | 44.5 | 44.3 | 42.0 | 40.9 | 2.3k | 19.3 GB |
| muse-glimmer-30b (reference, 08-12) | 51.2 | 50.8 | 48.8 | 46.6 | 3.4k | 19.0 GB |

Decode tok/s at batch 1, gen 256, 3 reps. Read it as:

- **It is a dense-27B-shaped decode curve, about 13% below Muse Glimmer.** Same
  quant family, same build, near-identical resident cost, and Qwen3.8 gives up
  roughly 6 tok/s at every context. The hybrid attention does not buy decode
  speed at batch 1 -- the 48 GDN layers replace attention that was already cheap
  at this scale, and the 16 full-attention layers still dominate.
- **It buys context flatness instead.** 44.5 → 40.9 tok/s from 512 to 32K is an
  8.1% falloff, against 8.9% for Muse Glimmer and 43% for the 3B-active MoE. The
  KV cache grows in a quarter of the layers, so at q8_0 it costs ~16 KiB/token
  rather than ~64. This is the reason to run the model, and 32K is nowhere near
  where it stops -- it is just where this sweep stops.
- **Prefill is the weak column.** 2.3k tok/s at 32K against 3.4k for Muse
  Glimmer, and TTFT at 32K is 11.2 s. A full-window cold prompt is expensive
  enough that prompt reuse is doing real work in an agent loop.

### Vision

| image | image tokens | prefill tok/s | TTFT | decode tok/s |
|---|---|---|---|---|
| 512x512 | 258 | 414 | 782 ms | 40.7 |
| 1024x1024 | 1026 | 876 | 1.26 s | 44.5 |

Same shape as Muse Glimmer -- expensive to ingest, free to reason over -- but
the image-token prefill rate is lower again (414-876 tok/s against 760-1237),
and the 512x512 cell decodes at 40.7 tok/s where the 1024 cell decodes at 44.5.
The small-image cell is short enough (322 total prompt tokens) that projector
encode is a visible fraction of the whole request, so read that row as a
per-request cost, not a per-token rate. The BF16 projector is the only one
published, as with Muse Glimmer.

### Two caveats on this table

Both are recorded rather than papered over, and both are cheap to clear on the
next run:

- The `quant` and `runtime` fields in `20260814.jsonl` are `unknown`. `llmbench`
  labels rows from the gateway config, and it was pointed at a config that did
  not yet carry the `qwen3.8-27b` entry this PR adds. The `arm` and
  `runtime_ver` fields were set on the command line and are correct.
- The measured server was running llama.cpp's default `repeat_penalty` of 1.1,
  not the model card's 1.0. The unit this PR ships pins it (confirmed via
  `/props`). Decode throughput is not meaningfully sensitive to a repetition
  penalty, so the table stands; any *quality* claim about these rows does not,
  because a penalised sampler is not the recipe.

### FP8 on vLLM: does not serve on this card

`Qwen/Qwen3.8-27B-FP8` is 28.77 GiB of weights against 22.07 GiB usable at
`--gpu-memory-utilization 0.92`, which is the practical ceiling here because the
Windows desktop compositor holds ~1.5 GiB of the 24 GiB (0.93 fails by 0.02
GiB). CPU offload is therefore mandatory, and it is bracketed by two independent
failures:

| `--cpu-offload-gb` | result |
|---|---|
| 8 | `ValueError: No available memory for the cache blocks` |
| 10 | same, including at `--max-model-len 8192` |
| 11 | inconclusive, see below |
| 12 | `ValueError: Pointer argument cannot be accessed from Triton (cpu tensor?)` |

The upper failure is the interesting one and it is structural. At 12 GiB the
offload reaches the Gated DeltaNet weights, and the GDN prefill kernel
(`flash_linear_attention/ops/fused_gdn_prefill_post_conv.py`, via
`vllm/model_executor/warmup/qwen_triton_warmup.py`) is a Triton kernel that
cannot address host memory. So offload cannot be raised arbitrarily to buy KV
room the way it can for a plain transformer: this model has a hard offload
ceiling somewhere below 12 GiB, and at that ceiling there is still not enough
VRAM left for a single cache block.

The 11 GiB row is honestly inconclusive rather than negative. That attempt ran
00:11-00:33, overlapping a second tenant on the same card (a GLM-OCR vLLM server
under `/root/kvant-ocr`, whose bench units started at 00:03, 00:24, 00:27 and
00:32), so its cache-block failure cannot be separated from the neighbour's VRAM
use. The 8, 10 and 12 GiB rows all ran before that tenant started and are clean.
Retesting 11 GiB on an idle card is the one experiment that would close this
out; the bracket around it makes success unlikely but not impossible.

The checkout stays on the box as a quality reference. Its gateway entry is
present and deliberately not autostarted. Note also that this checkout is
sharded as `outside.safetensors` plus `layers-0..63.safetensors`, not the usual
`model-0000N-of-0000M` plus `index.json`, so a download loop written against the
index scheme finds nothing and exits successfully having fetched no weights.

## Solve rate

From the Mac (needs Docker, git, uv, and a built tomo image: `cd $LABS_DIR && lab build tomo`):

```
LABS_DIR=~/github/tamnd/tomo-labs TOKEN=<token> \
  MODELS=qwen3-32b,qwen3-30b-a3b-exl3,... \
  scripts/solve-swebench.sh
```

The gateway base has no `/v1` suffix: the trace proxy appends the incoming path, which already carries `/v1`.
