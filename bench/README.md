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

Flags: `-base-url` (default the box tailnet gateway), `-config` (labels each row's runtime and quant from the gateway config), `-contexts`, `-gen`, `-reps`, `-out`. `-dry-run` prints the plan without calling the endpoint.

## Muse Glimmer 30B (2026-08-10)

First numbers for Meta's Muse Glimmer 30B, Unsloth `UD-Q4_K_XL` on llama-server
b10355 (WSL2, CUDA 13.1, sm_89). Rows are in `results/20260810.jsonl`; the
reference arm is the box's current `chat` default on the same sweep and the same
day. `qwen3-32b` is carried from `results/20260719.jsonl` as the dense yardstick.

| model | ctx 512 | 2048 | 8192 | 32768 | prefill @32K | VRAM |
|---|---|---|---|---|---|---|
| muse-glimmer-30b | 50.8 | 50.5 | 49.2 | 46.6 | 33.8k tok/s | 15.1 GB |
| qwen3-30b-a3b (MoE, ollama) | 212.5 | 205.9 | 183.5 | 121.5 | 38.6k tok/s | 18.9 GB |
| qwen3-32b (dense, ollama) | 41.9 | 41.2 | 38.7 | - | 9.9k tok/s | 21.2 GB |

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
  Silicon only. Hence the llama-server path.

## Solve rate

From the Mac (needs Docker, git, uv, and a built tomo image: `cd $LABS_DIR && lab build tomo`):

```
LABS_DIR=~/github/tamnd/tomo-labs TOKEN=<token> \
  MODELS=qwen3-32b,qwen3-30b-a3b-exl3,... \
  scripts/solve-swebench.sh
```

The gateway base has no `/v1` suffix: the trace proxy appends the incoming path, which already carries `/v1`.
