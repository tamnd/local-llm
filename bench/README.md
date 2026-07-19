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

## Solve rate

From the Mac (needs Docker, git, uv, and a built tomo image: `cd $LABS_DIR && lab build tomo`):

```
LABS_DIR=~/github/tamnd/tomo-labs TOKEN=<token> \
  MODELS=qwen3-32b,qwen3-30b-a3b-exl3,... \
  scripts/solve-swebench.sh
```

The gateway base has no `/v1` suffix: the trace proxy appends the incoming path, which already carries `/v1`.
