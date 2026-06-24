#!/usr/bin/env bash
# bench-compare.sh: measure decode tok/s for the same model served by Ollama,
# llama.cpp inproc (via the gateway), and vLLM (via the gateway). Produces a
# tab-separated table at the end so the numbers can go into spec 18 directly.
#
# Each backend must already be loaded and answering requests before this script
# runs. For inproc that means the gateway started with -tags llama and the model
# configured. For vLLM that means the relevant systemd unit is active. For Ollama
# it just needs to be running (it loads on demand).
#
# Usage:
#   GATEWAY_TOKEN=<token> ./scripts/bench-compare.sh [model-key]
#
# model-key defaults to "qwen3-35b". The key must match an entry in the gateway
# config. Use comma-separated keys to run multiple: "qwen3-35b,gpt-oss-20b".
#
# The Ollama URL and the gateway URL can be overridden with:
#   OLLAMA_URL=http://127.0.0.1:11434
#   GATEWAY_URL=http://127.0.0.1:8888
#
# Requires: curl, python3 (for JSON parsing), nvidia-smi.

set -euo pipefail

OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
GATEWAY_URL="${GATEWAY_URL:-http://127.0.0.1:8888}"
GATEWAY_TOKEN="${GATEWAY_TOKEN:-}"
MODELS="${1:-qwen3-35b}"
RUNS="${BENCH_RUNS:-5}"          # measured generations per model per backend
NUM_PREDICT="${NUM_PREDICT:-256}" # decode tokens per run

if [ -z "${GATEWAY_TOKEN}" ]; then
    echo "GATEWAY_TOKEN is required. Export it before running." >&2
    exit 1
fi

# A ~300-token prompt so prompt-processing time is comparable across models.
PROMPT="You are a careful systems engineer. Explain step by step how a write-ahead
log guarantees durability and crash recovery in a database engine. Cover the log
record format, the checkpoint mechanism, redo and undo phases, group commit, and
the fsync ordering that makes the whole scheme correct. Then describe how this
interacts with page caching and how a buffer pool decides which dirty pages to
flush. Be concrete and use examples where it helps."

vram_mb() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
        | awk '{print int($1)}'
}

# Warmup: one throw-away generation to load the model before measuring.
ollama_run() {
    local model="$1" n="$2"
    curl -sf -X POST "${OLLAMA_URL}/api/generate" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${model}\",\"prompt\":$(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "${PROMPT}"),\"stream\":false,\"options\":{\"num_predict\":${n},\"temperature\":0}}"
}

gateway_run() {
    local model="$1" n="$2"
    curl -sf -X POST "${GATEWAY_URL}/v1/chat/completions" \
        -H "Authorization: Bearer ${GATEWAY_TOKEN}" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${model}\",\"messages\":[{\"role\":\"user\",\"content\":$(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "${PROMPT}")}],\"stream\":false,\"max_tokens\":${n},\"temperature\":0}"
}

bench_ollama() {
    local model="$1"
    echo "  [ollama] warmup..." >&2
    ollama_run "${model}" 16 >/dev/null 2>&1 || true
    local sum=0 count=0
    for _ in $(seq 1 "${RUNS}"); do
        local r
        r=$(ollama_run "${model}" "${NUM_PREDICT}" 2>/dev/null) || continue
        local ns count_tok
        ns=$(echo "${r}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('eval_duration',0))")
        count_tok=$(echo "${r}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('eval_count',0))")
        if [ "${ns}" -gt 0 ] && [ "${count_tok}" -gt 0 ]; then
            local tps
            tps=$(python3 -c "print(round(${count_tok} / (${ns} / 1e9), 1))")
            sum=$(python3 -c "print(${sum} + ${tps})")
            count=$((count + 1))
        fi
    done
    if [ "${count}" -gt 0 ]; then
        python3 -c "print(round(${sum} / ${count}, 1))"
    else
        echo "N/A"
    fi
}

bench_gateway() {
    local model="$1" backend_label="$2"
    echo "  [${backend_label}] warmup..." >&2
    gateway_run "${model}" 16 >/dev/null 2>&1 || true
    local sum=0 count=0
    for _ in $(seq 1 "${RUNS}"); do
        local r t0 t1 elapsed count_tok
        t0=$(date +%s%3N)
        r=$(gateway_run "${model}" "${NUM_PREDICT}" 2>/dev/null) || continue
        t1=$(date +%s%3N)
        elapsed=$(python3 -c "print(${t1} - ${t0})")
        count_tok=$(echo "${r}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('usage',{}).get('completion_tokens',0))")
        if [ "${elapsed}" -gt 0 ] && [ "${count_tok}" -gt 0 ]; then
            local tps
            tps=$(python3 -c "print(round(${count_tok} / (${elapsed} / 1000), 1))")
            sum=$(python3 -c "print(${sum} + ${tps})")
            count=$((count + 1))
        fi
    done
    if [ "${count}" -gt 0 ]; then
        python3 -c "print(round(${sum} / ${count}, 1))"
    else
        echo "N/A"
    fi
}

printf "\n%-20s  %10s  %10s  %10s  %10s\n" \
    "model" "ollama" "inproc" "vllm" "vram_mb"
printf "%s\n" "$(printf '%.0s-' {1..70})"

IFS=',' read -ra MODEL_LIST <<< "${MODELS}"
for model in "${MODEL_LIST[@]}"; do
    model="${model// /}"
    echo "benchmarking ${model}..." >&2

    # Map gateway model key to Ollama model name. The gateway config's
    # upstream_model field holds the Ollama name; we hard-code the common ones
    # here and let the variable be overridden.
    case "${model}" in
        qwen3-35b)   ollama_model="${OLLAMA_MODEL:-qwen3.6:35b}" ;;
        qwen3-27b)   ollama_model="${OLLAMA_MODEL:-qwen3.6:27b}" ;;
        gpt-oss-20b) ollama_model="${OLLAMA_MODEL:-gpt-oss:20b}" ;;
        *)           ollama_model="${OLLAMA_MODEL:-${model}}" ;;
    esac

    vram_before=$(vram_mb)

    tps_ollama=$(bench_ollama "${ollama_model}")
    vram_ollama=$(vram_mb)

    tps_inproc=$(bench_gateway "${model}-inproc" "inproc" 2>/dev/null || echo "N/A")
    tps_vllm=$(bench_gateway "${model}-vllm" "vllm" 2>/dev/null || echo "N/A")

    vram_peak=$(vram_mb)
    vram_net=$((vram_peak - vram_before))

    printf "%-20s  %10s  %10s  %10s  %10d\n" \
        "${model}" "${tps_ollama}" "${tps_inproc}" "${tps_vllm}" "${vram_net}"
done
echo ""
echo "tok/s = decode tokens per second (higher is better)"
echo "vram_mb = net VRAM used by the last model loaded"
