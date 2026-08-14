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

# Python 3.12 is the floor, and it is a hard one. This script used to pin 3.11,
# which was right for vLLM 0.9.x, but the flashinfer that vLLM 0.27.x depends on
# has a module-scope annotation of array.array[int] in comm/fd_exchange.py, and
# array.array only became subscriptable in Python 3.12. On 3.11 the import raises
#
#   TypeError: type 'array.array' is not subscriptable
#
# from inside EngineCore startup -- after the architecture has resolved, the FP8
# kernels have been selected and the attention backend has been chosen. Every log
# line up to that point looks healthy, so the failure reads as a model or a VRAM
# problem rather than an interpreter version problem. It is neither.
#
# The upper bound is still real: do not move to 3.14, whose tokenizers C
# extension is not built.
if ! python3.12 --version &>/dev/null; then
    echo "installing python3.12 via deadsnakes"
    apt-get update -q
    apt-get install -y -q software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update -q
    apt-get install -y -q python3.12 python3.12-venv python3.12-dev
fi

# An existing venv built on an older interpreter has to be recreated, not
# upgraded in place: the wheels under it are ABI-tagged for the interpreter that
# built it, and pip will happily leave a 3.11 torch sitting next to a 3.12
# python.
if [ -f "${PY}" ] && ! "${PY}" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,12) else 1)'; then
    echo "existing venv is $(${PY} --version 2>&1); rebuilding on 3.12"
    rm -rf "${VENV}"
fi
if [ ! -f "${PY}" ]; then
    echo "creating venv at ${VENV}"
    python3.12 -m venv "${VENV}"
fi

# Upgrade pip before anything else.
"${PIP}" install --quiet --upgrade pip setuptools

# Build/unpack scratch goes to a real disk. /tmp in this distro is a 16 GB tmpfs
# (RAM-backed) that is routinely most-full with other work, and a vLLM install
# unpacks well over 10 GB of wheels through it: the failure is
# "OSError: [Errno 28] No space left on device" while df on / shows hundreds of
# gigabytes free, which reads as nonsense until you notice which filesystem the
# unpack is actually on. --no-cache-dir keeps pip from holding a second copy.
export TMPDIR="${TMPDIR_OVERRIDE:-/root/tmp}"
mkdir -p "${TMPDIR}"
PIP_INSTALL=("${PIP}" install --quiet --no-cache-dir)

# vLLM 0.27.1 pulls torch 2.13.0. As of torch 2.9+, the PyPI manylinux_2_28
# wheel includes CUDA dispatch (no separate CUDA extra index needed); the wheel
# links against the system CUDA at runtime. Let vLLM's own pin resolve torch
# rather than pinning it separately here -- the previous exact torch==2.11.0 pin
# had to be edited in lockstep with every vLLM bump and silently conflicts if it
# drifts.
#
# vLLM 0.27.1 is the current stable release and the first to carry
# Qwen3_5ForConditionalGeneration, which is the architecture Qwen3.8 reports in
# config.json. 0.23.0 rejects the checkpoint at load with an unknown-architecture
# error, so this is a hard floor, not a preference.
#
# 0.27.2 (nightly at time of writing) is only needed for the MTP speculative
# decoding fixes. Qwen3.8 ships a trained MTP head and vLLM exposes it as
# Qwen3_5MTP; this script does not enable it, so stable is sufficient.
if ! "${PY}" -c "import vllm; v=vllm.__version__; assert v.startswith('0.27')" 2>/dev/null; then
    echo "installing vllm 0.27.1 (pulls torch 2.13.x)"
    "${PIP_INSTALL[@]}" vllm==0.27.1
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
#   8100 - qwen3.8-27b-fp8    -> Qwen/Qwen3.8-27B-FP8 (native FP8, 30.89 GB, does NOT fit)
#   8101 - gpt-oss-20b        -> openai/gpt-oss-20b (native MXFP4 MoE, 12.8 GiB)
#
# 8102 (qwen3-32b-awq) and 8103 (qwen3-30b-a3b-awq) were retired along with their
# weights and gateway entries to make room for Qwen3.8.
#
# Only one model can run at a time on a single 24 GB GPU. The gateway uses
# hot_swap to unload the active model before loading the next one.
#
# HF_HUB_OFFLINE=1 is required. Without it huggingface_hub's _detect_agent
# module tries a registry TCP connection during import; vLLM's _interrupt_init
# SIGINT handler patches socket.connect and the connection raises
# KeyboardInterrupt("terminated"), killing the service in ~16 s before any
# weights load. With local paths + offline mode the startup is purely local.
#
# Pre-download weights using wget/curl (faster than snapshot_download on HF XET
# CDN). Qwen3.8-27B-FP8 does NOT use the model-0000N-of-0000M plus
# model.safetensors.index.json layout every other checkout here uses: it shards
# as outside.safetensors plus layers-0..layers-63.safetensors. A loop written
# against the index scheme downloads the config files, finds no shards, and exits
# successfully having fetched nothing.
#
# Verify by size, not by presence. A long HF/2 transfer can die with
# "HTTP/2 stream 1 was not closed cleanly: CANCEL (err 8)" partway through a
# shard, and a name-only check then treats the truncated file as done. --http1.1
# avoids the CANCEL; comparing against the remote content-length catches it when
# it happens anyway.
#
#   D=/models/hf/Qwen3.8-27B-FP8; mkdir -p $D; cd $D
#   HF=https://huggingface.co/Qwen/Qwen3.8-27B-FP8/resolve/main
#   for f in config.json generation_config.json tokenizer.json tokenizer_config.json \
#             vocab.json merges.txt chat_template.jinja preprocessor_config.json \
#             video_preprocessor_config.json; do
#       curl -fsSL --retry 5 -o "$f" "$HF/$f"
#   done
#   for f in outside.safetensors $(for i in $(seq 0 63); do echo layers-$i.safetensors; done); do
#       want=$(curl -fsSLI "$HF/$f" | tr -d '\r' | awk 'tolower($1)=="content-length:"{n=$2} END{print n}')
#       [ "$(stat -c %s "$f" 2>/dev/null || echo 0)" = "$want" ] && continue
#       curl -fsSL --http1.1 --retry 5 --retry-all-errors -C - -o "$f" "$HF/$f"
#   done
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
KillMode=control-group
ExecStart=${PY} -m vllm.entrypoints.openai.api_server \
    --model ${local_path} \
    --host 127.0.0.1 \
    --port ${port} \
    --gpu-memory-utilization ${mem} \
    --tensor-parallel-size 1 \
    --enable-chunked-prefill \
    ${extra_flags}
Restart=on-failure
RestartSec=30
StandardOutput=journal
StandardError=journal
SyslogIdentifier=vllm-${name}

[Install]
WantedBy=multi-user.target
UNIT
    echo "wrote /etc/systemd/system/vllm-${name}.service"
}

# 0.92 is the practical ceiling for gpu-memory-utilization on this box, not a
# round number picked for taste. The Windows desktop and the WDDM driver hold
# roughly 1.5 GiB of the 24 GB card at all times, so vLLM sees 22.45 of 23.99
# GiB free at startup and refuses to start if the requested fraction exceeds it:
#
#   ValueError: Free memory on device cuda:0 (22.45/23.99 GiB) on startup is
#   less than desired GPU memory utilization (0.95, 22.79 GiB).
#
# gpt-oss-20b MXFP4 weights total ~20 GB (U8 + BF16 embeddings). It fits at 0.92
# (22.07 GiB effective) with a shorter context window.
write_unit "gpt-oss-20b" "${VLLM_MODEL_DIR}/gpt-oss-20b" 8101 "0.92" "--kv-cache-dtype fp8 --max-model-len 8192"

# Qwen3.8-27B-FP8 DOES NOT START on this card. The unit is written so the
# failure is reproducible, not because it serves; the 4-bit GGUF arm in
# scripts/11-llamacpp-qwen38.sh is the serving path for this model.
#
# 28.77 GiB of weights against 22.07 GiB of usable card means CPU offload is
# mandatory, and it is bracketed by two independent failures:
#
#   --cpu-offload-gb 8, 10 -> "No available memory for the cache blocks", even at
#                             --max-model-len 4096 --max-num-seqs 2.
#   --cpu-offload-gb 12    -> offload reaches the Gated DeltaNet weights and the
#                             GDN prefill warmup, a Triton kernel, cannot address
#                             host memory: "Pointer argument cannot be accessed
#                             from Triton (cpu tensor?)".
#
# Unlike a plain transformer this model cannot trade more offload for more KV
# room, because past ~12 GiB the offloaded tensors are ones Triton must touch.
# 11 is the only value not yet ruled out and is what the unit carries; the single
# attempt at it overlapped another tenant on the card and proved nothing.
#
# Host RAM is the binding prerequisite before retrying: startup peaks near 39 GB
# RSS, so WSL2 needs its .wslconfig memory ceiling well above the default
# half-of-host. --enforce-eager because CUDA graph capture over a partially
# offloaded model is where this falls over first.
write_unit "qwen3.8-27b-fp8" "${VLLM_MODEL_DIR}/Qwen3.8-27B-FP8" 8100 "0.92" "--kv-cache-dtype fp8 --max-model-len 8192 --cpu-offload-gb 11 --enforce-eager"

systemctl daemon-reload
echo ""
echo "Units written. Download weights first (see comment block above write_unit), then:"
echo "  systemctl start vllm-gpt-oss-20b"
echo "  systemctl start vllm-qwen3.8-27b-fp8   # expected to fail; see the note above"
echo ""
echo "Only one runs at a time on a 24 GB card; the gateway unloads the active"
echo "model before loading the next. Then edit configs/llmgw.yaml."
