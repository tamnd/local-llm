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

## Solve rate

From the Mac (needs Docker, git, uv, and a built tomo image: `cd $LABS_DIR && lab build tomo`):

```
LABS_DIR=~/github/tamnd/tomo-labs TOKEN=<token> \
  MODELS=qwen3-32b,qwen3-30b-a3b-exl3,... \
  scripts/solve-swebench.sh
```

The gateway base has no `/v1` suffix: the trace proxy appends the incoming path, which already carries `/v1`.
