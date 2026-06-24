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

# Upgrade pip before anything else; older pip misreads the CUDA extra index.
"${PIP}" install --quiet --upgrade pip

# PyTorch 2.7 + CUDA 12.6. The CUDA extra index supplies the cu126 wheel so
# we avoid the generic cpu wheel from PyPI.
if ! "${PY}" -c "import torch; assert '2.7' in torch.__version__" 2>/dev/null; then
    echo "installing pytorch 2.7 cu126"
    "${PIP}" install \
        torch==2.7.0+cu126 \
        --index-url https://download.pytorch.org/whl/cu126 \
        --quiet
fi

# vLLM 0.9.1 with FlashInfer 0.2.x for fused MoE, chunked prefill, and FP8
# KV cache on Ada (sm_89). flashinfer prebuilt wheel for cu126 + torch 2.7.
if ! "${PY}" -c "import vllm" 2>/dev/null; then
    echo "installing vllm 0.9.1"
    "${PIP}" install vllm==0.9.1 --quiet

    # FlashInfer provides optimized MoE routing and attention backends. Without
    # it vLLM falls back to triton kernels which are slower on Ada.
    echo "installing flashinfer 0.2.x"
    "${PIP}" install flashinfer-python \
        --index-url https://flashinfer.ai/whl/cu126/torch2.7 \
        --quiet
fi

# huggingface_hub CLI for model downloads. hf_transfer speeds up large pulls.
"${PIP}" install --quiet huggingface_hub hf_transfer

echo "vLLM $(${PY} -c 'import vllm; print(vllm.__version__)') installed at ${VENV}"

# Create the model directory if it does not exist. Models are large; this is
# typically a symlink to a drive with enough space.
mkdir -p "${VLLM_MODEL_DIR}"

# Write systemd units for each model that should run as a persistent service.
# The gateway expects each vLLM model on a fixed loopback port. Override the
# HF_HOME so all units share the cache. Each unit depends on
# network-online.target so Tailscale is up before the gateway starts.
#
# Port assignments (matching configs/llmgw.yaml):
#   8100 - qwen3.6:35b (Qwen3-30B-A3B-Instruct) - MoE path
#   8101 - gpt-oss:20b (openai-community/gpt-oss-20b) - MoE path

write_unit() {
    local name="$1" model="$2" port="$3" mem="$4"
    cat > "/etc/systemd/system/vllm-${name}.service" <<UNIT
[Unit]
Description=vLLM ${name} on port ${port}
After=network-online.target nvidia-persistenced.service
Wants=network-online.target

[Service]
Type=simple
Environment=HF_HOME=${VLLM_MODEL_DIR}/.cache/huggingface
Environment=HF_HUB_ENABLE_HF_TRANSFER=1
Environment=CUDA_VISIBLE_DEVICES=0
ExecStart=${PY} -m vllm.entrypoints.openai.api_server \\
    --model ${model} \\
    --host 127.0.0.1 \\
    --port ${port} \\
    --gpu-memory-utilization ${mem} \\
    --tensor-parallel-size 1 \\
    --max-model-len 32768 \\
    --enable-chunked-prefill \\
    --enforce-eager
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

write_unit "qwen3-35b"  "Qwen/Qwen3-30B-A3B-Instruct"            8100 "0.85"
write_unit "gpt-oss-20b" "openai-community/gpt-oss-20b"           8101 "0.70"

systemctl daemon-reload
echo ""
echo "Units written. To start a backend:"
echo "  systemctl start vllm-qwen3-35b"
echo "  systemctl start vllm-gpt-oss-20b"
echo ""
echo "On first start vLLM downloads the weights (~18 GB each)."
echo "Set HF_TOKEN if the model is gated: export HF_TOKEN=hf_..."
echo ""
echo "Then edit configs/llmgw.yaml and uncomment the vllm entries."
