#!/usr/bin/env bash
# 08-vllm.sh: install vLLM in WSL2 for the RTX 4090 box and register a
# systemd service that starts it on boot (spec 2065 doc 08 section 5.4).
#
# vLLM is the Option B inference backend: where llama.cpp inproc wins on
# dense models (+59% vs Ollama), vLLM's fused MoE CUDA kernels (Marlin/
# FlashInfer) target better throughput on MoE-heavy models like qwen3.6:35b
# and gpt-oss:20b. The gateway routes each model to the backend defined in
# configs/llmgw.yaml; no changes to the clients.
#
# Requirements:
#   - WSL2 Ubuntu 24.04+ on the RTX 4090 box
#   - CUDA 12.x already installed (script 02 runs this via install_cuda12.sh)
#   - Python 3.11 (vLLM 0.9+ requires 3.9-3.12; system 3.14 is too new)
#   - At least 60 GB free disk for weights under VLLM_MODEL_DIR
#   - VLLM_MODEL_DIR set to the directory where HF weights should land
#     (default /models/hf; symlink Ollama blobs to .gguf files inside it
#     to skip re-downloading for GGUF-backed models)
#
# Run once, then reload llmgw with the vllm backend entries uncommented in
# configs/llmgw.yaml. Models are served on loopback ports 8100+ so only the
# gateway can reach them.

set -euo pipefail

VLLM_MODEL_DIR="${VLLM_MODEL_DIR:-/models/hf}"
VENV="/opt/vllm-venv"
PY="${VENV}/bin/python"
PIP="${VENV}/bin/pip"

# Python 3.11 is required. vLLM 0.9.x does not support 3.14 (tokenizers C
# extension is not built for it yet). Install deadsnakes if not present.
if ! python3.11 --version &>/dev/null; then
    echo "installing python3.11 via deadsnakes"
    apt-get update -q
    apt-get install -y -q software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update -q
    apt-get install -y -q python3.11 python3.11-venv python3.11-dev
fi

if [ ! -f "${PY}" ]; then
    echo "creating venv at ${VENV}"
    python3.11 -m venv "${VENV}"
fi

# Upgrade pip before anything else.
"${PIP}" install --quiet --upgrade pip setuptools

# vLLM 0.23.0 requires torch==2.11.0 exactly. As of torch 2.9+, the PyPI
# manylinux_2_28 wheel includes CUDA dispatch (no separate CUDA extra index
# needed); the wheel links against the system CUDA at runtime. Install torch
# first so vLLM's constraint resolves cleanly.
if ! "${PY}" -c "import torch; assert '2.11' in torch.__version__" 2>/dev/null; then
    echo "installing pytorch 2.11.0"
    "${PIP}" install torch==2.11.0 --quiet
fi

# vLLM 0.23.0: released 2026-06-15, adds gpt-oss-20b support, Qwen3.6 MoE,
# and FlashInfer 0.6.x fused MoE kernels on Ada (sm_89).
# FlashInfer 0.6.12 is a hard dep pinned in vllm's wheel; pip resolves it.
if ! "${PY}" -c "import vllm; v=vllm.__version__; assert v.startswith('0.23')" 2>/dev/null; then
    echo "installing vllm 0.23.0"
    "${PIP}" install vllm==0.23.0 --quiet
fi

# huggingface_hub CLI for model downloads. hf_transfer speeds up large pulls.
"${PIP}" install --quiet huggingface_hub hf_transfer

echo "vLLM $(${PY} -c 'import vllm; print(vllm.__version__)') torch $(${PY} -c 'import torch; print(torch.__version__)') installed at ${VENV}"

# Create the model directory if it does not exist. Models are large; this is
# typically a symlink to a drive with enough space.
mkdir -p "${VLLM_MODEL_DIR}"

# Write systemd units for each model that should run as a persistent service.
# The gateway expects each vLLM model on a fixed loopback port. Override the
# HF_HOME so all units share the cache. Each unit depends on
# network-online.target so Tailscale is up before the gateway starts.
#
# Port assignments (matching configs/llmgw.yaml):
#   8100 - qwen3.6:35b  -> Qwen/Qwen3.6-35B-A3B-FP8 (FP8 quant, fits in 24 GB)
#   8101 - gpt-oss:20b  -> openai/gpt-oss-20b (native MXFP4 MoE)
#
# Qwen3.6-35B-A3B uses gated-delta-network SSM layers; set --max-num-seqs=512
# to keep the recurrent state cache from exceeding VRAM (default 1024 is too
# large for a 24 GB card with FP8 weights already resident).
#
# HF_HUB_OFFLINE=1 is required. Without it huggingface_hub's _detect_agent
# module tries a registry TCP connection during import; vLLM's _interrupt_init
# SIGINT handler patches socket.connect and the connection raises
# KeyboardInterrupt("terminated"), killing the service in ~16 s before any
# weights load. With local paths + offline mode the startup is purely local.
#
# Pre-download weights using wget (faster than snapshot_download on HF XET CDN):
#   mkdir -p /models/hf/Qwen3-14B-FP8
#   cd /models/hf/Qwen3-14B-FP8
#   HF=https://huggingface.co/Qwen/Qwen3-14B-FP8
#   for f in config.json generation_config.json tokenizer.json tokenizer_config.json \
#             vocab.json merges.txt model.safetensors.index.json; do
#       wget -q -nc "$HF/resolve/main/$f"
#   done
#   for i in 1 2 3 4; do
#       wget -q -nc -c "$HF/resolve/main/model-0000${i}-of-00004.safetensors" &
#   done; wait
#
#   mkdir -p /models/hf/gpt-oss-20b
#   cd /models/hf/gpt-oss-20b
#   HF2=https://huggingface.co/openai/gpt-oss-20b
#   for f in config.json generation_config.json tokenizer.json tokenizer_config.json \
#             special_tokens_map.json chat_template.jinja model.safetensors.index.json; do
#       wget -q -nc "$HF2/resolve/main/$f"
#   done
#   for i in 0 1 2; do
#       wget -q -nc -c "$HF2/resolve/main/model-0000${i}-of-00002.safetensors" &
#   done; wait

write_unit() {
    local name="$1" local_path="$2" port="$3" mem="$4" extra_flags="${5:-}"
    cat > "/etc/systemd/system/vllm-${name}.service" <<UNIT
[Unit]
Description=vLLM ${name} on port ${port}
After=network-online.target nvidia-persistenced.service
Wants=network-online.target

[Service]
Type=simple
Environment=HF_HOME=${VLLM_MODEL_DIR}/.cache/huggingface
Environment=HF_HUB_OFFLINE=1
Environment=CUDA_VISIBLE_DEVICES=0
Environment="PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=${PY} -m vllm.entrypoints.openai.api_server \
    --model ${local_path} \
    --host 127.0.0.1 \
    --port ${port} \
    --gpu-memory-utilization ${mem} \
    --tensor-parallel-size 1 \
    --enable-chunked-prefill \
    ${extra_flags}
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=vllm-${name}

[Install]
WantedBy=multi-user.target
UNIT
    echo "wrote /etc/systemd/system/vllm-${name}.service"
}

# Qwen3.6-27B-FP8 (30.9 GB) and Qwen3.6-35B-A3B-FP8 (~27 GB) both exceed the
# 24 GB VRAM on the RTX 4090. Use Qwen3-14B-FP8 (16.3 GB) instead — same
# architecture family, fits with room for KV cache.
#
# gpt-oss-20b MXFP4 weights total ~20 GB (U8 + BF16 embeddings). It barely
# fits at 0.92 utilization (22.1 GB effective) with a shorter context window.
write_unit "qwen3-14b-fp8" "${VLLM_MODEL_DIR}/Qwen3-14B-FP8" 8100 "0.85" "--max-model-len 16384"
write_unit "gpt-oss-20b"   "${VLLM_MODEL_DIR}/gpt-oss-20b"   8101 "0.92" "--kv-cache-dtype fp8 --max-model-len 8192"

systemctl daemon-reload
echo ""
echo "Units written. Download weights first (see comment block above write_unit), then:"
echo "  systemctl start vllm-qwen3-14b-fp8"
echo "  systemctl start vllm-gpt-oss-20b"
echo ""
echo "Then edit configs/llmgw.yaml and uncomment the vllm entries."
