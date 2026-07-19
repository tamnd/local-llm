#!/usr/bin/env bash
# solve-swebench.sh: the real-task solve runner. It drives the tomo-labs
# swebench-live harness against the llmgw gateway once per backend model, so the
# same real GitHub-issue tasks (graded by hidden fail_to_pass tests, not a smoke
# echo) run through ollama, vLLM, and TabbyAPI/ExLlamaV3 in turn. For each model
# it records solve rate, tokens, and model rounds as one JSON line in
# bench/results/solve-<date>.jsonl, the correctness companion to the throughput
# rows cmd/llmbench writes.
#
# It reuses the harness's own aggregator (lab report --json) rather than
# reparsing result.json: the report keys by tool and keeps the latest run per
# scenario, so a snapshot taken right after each model's sequential run reflects
# that model. Runs must be sequential for that reason; do not parallelize.
#
# Wiring note that is easy to get wrong: the trace proxy forwards to
# $UPSTREAM + the incoming request path, and the incoming path already carries
# /v1 (tomo talks to the proxy at .../v1/chat/completions). So LAB_UPSTREAM is
# the gateway base WITHOUT a /v1 suffix.
#
# Prerequisites on the host running this (the Mac is fine):
#   - Docker, git, and uv, for the tomo-labs harness and its grading venvs.
#   - The tomo tool image built once: (cd $LABS_DIR && lab build tomo).
#   - The gateway reachable at $GATEWAY, serving every model in $MODELS.
#
# Env:
#   LABS_DIR   path to the tomo-labs checkout        (required)
#   GATEWAY    gateway base, no /v1 suffix           (default http://100.71.238.128:8888)
#   TOKEN      gateway bearer token                  (or LLMBENCH_TOKEN)
#   MODELS     comma-separated gateway model ids     (required)
#   SUITE      eval suite name                       (default swebench-live)
#   TASKS      comma-separated scenario ids to run   (default: the whole suite)
#              A local 4-bit 32B decodes at tens of tok/s, and an agentic task
#              burns tens of thousands of tokens across many rounds, so the full
#              15-task suite times six backends is hours of wall clock. TASKS
#              narrows the run to a named subset; the harness runs one scenario
#              per `lab run tomo <id>` when an id is given.
#   CONFIG     llmgw config, for the runtime label   (default configs/llmgw.yaml)
#   OUT        output JSONL path                     (default bench/results/solve-<date>.jsonl)
set -euo pipefail

GATEWAY="${GATEWAY:-http://100.71.238.128:8888}"
TOKEN="${TOKEN:-${LLMBENCH_TOKEN:-}}"
SUITE="${SUITE:-swebench-live}"
CONFIG="${CONFIG:-configs/llmgw.yaml}"
DATE="$(date +%Y%m%d)"
OUT="${OUT:-bench/results/solve-${DATE}.jsonl}"

die() { echo "solve-swebench: $*" >&2; exit 1; }

[ -n "${LABS_DIR:-}" ] || die "LABS_DIR is required (path to the tomo-labs checkout)"
[ -d "$LABS_DIR" ] || die "LABS_DIR $LABS_DIR is not a directory"
[ -n "${MODELS:-}" ] || die "MODELS is required (comma-separated gateway model ids)"
[ -n "$TOKEN" ] || die "TOKEN (or LLMBENCH_TOKEN) is required"
command -v jq >/dev/null || die "jq is required to fold the reports"

# runtime_of prints the backend a model is served by, read from the gateway
# config, so a row is self-describing (runtime=tabby) without a second list to
# keep in sync. Falls back to unknown if the model or config is absent.
runtime_of() {
	local model="$1"
	if [ -f "$CONFIG" ]; then
		awk -v m="$model" '
			$0 ~ "^  "m":" {inm=1; next}
			inm && /^  [a-zA-Z0-9_-]+:/ {inm=0}
			inm && /backend:/ {gsub(/.*backend:[ ]*/,""); gsub(/[ "]/,""); print; exit}
		' "$CONFIG"
	fi
}

mkdir -p "$(dirname "$OUT")"
echo "solve: suite=$SUITE gateway=$GATEWAY out=$OUT" >&2

IFS=',' read -ra MODEL_LIST <<< "$MODELS"
for raw in "${MODEL_LIST[@]}"; do
	model="$(echo "$raw" | xargs)"
	[ -n "$model" ] || continue
	runtime="$(runtime_of "$model")"; runtime="${runtime:-unknown}"
	echo "==== $model (runtime=$runtime) ====" >&2

	# Point the harness at the gateway for this model, then run the suite (or the
	# named TASKS subset). The proxy inside each task container forwards to
	# $GATEWAY; the key is passed straight through as the bearer token.
	(
		cd "$LABS_DIR"
		if [ -n "${TASKS:-}" ]; then
			IFS=',' read -ra TASK_LIST <<< "$TASKS"
			for t in "${TASK_LIST[@]}"; do
				task="$(echo "$t" | xargs)"; [ -n "$task" ] || continue
				echo "-- $model / $task --" >&2
				LAB_UPSTREAM="$GATEWAY" LAB_MODEL="$model" OPENCODE_API_KEY="$TOKEN" \
					go run ./cmd/lab run tomo "$task" --suite "$SUITE"
			done
		else
			LAB_UPSTREAM="$GATEWAY" LAB_MODEL="$model" OPENCODE_API_KEY="$TOKEN" \
				go run ./cmd/lab run tomo --suite "$SUITE"
		fi
	)

	# Snapshot this model's aggregate before the next model's run overwrites the
	# latest-per-scenario rows the report reads.
	report="$(cd "$LABS_DIR" && go run ./cmd/lab report --suite "$SUITE" --json)"
	echo "$report" | jq -c --arg date "$DATE" --arg model "$model" \
		--arg runtime "$runtime" --arg suite "$SUITE" '
		map(select(.tool == "tomo")) | .[0] // {} |
		{
			date: $date, model: $model, runtime: $runtime, suite: $suite,
			runs: (.runs // 0), passed: (.passed // 0),
			solve_rate: (if (.runs // 0) > 0 then ((.passed // 0) / .runs) else 0 end),
			total_tokens: (.total_tokens // 0),
			avg_model_calls: (.avg_model_calls // 0),
			total_cost_usd: (.total_cost_usd // 0),
			stream_fail_runs: (.stream_fail_runs // 0)
		}' >> "$OUT"
done

echo "done -> $OUT" >&2
echo >&2
echo "model                 runtime   solved  tokens     rounds  cost\$" >&2
jq -r '. | [.model, .runtime, "\(.passed)/\(.runs)", (.total_tokens|tostring), (.avg_model_calls|tostring), (.total_cost_usd|tostring)] | @tsv' "$OUT" |
	awk -F'\t' '{printf "%-20s  %-8s  %-6s  %-9s  %-6s  %s\n", $1,$2,$3,$4,$5,$6}' >&2
