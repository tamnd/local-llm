#!/usr/bin/env bash
# 11-llamacpp-qwen38.sh: bring the shared WSL2 llama.cpp checkout up to master
# and stage the Qwen3.8-27B GGUF set, so the gateway's qwen3.8-27b entry has a
# server to adopt.
#
# The served configuration follows Unsloth's run page for this model
# (https://unsloth.ai/docs/models/qwen3.8) and Qwen's own card: the UD-Q4_K_XL
# dynamic quant, the Thinking sampling preset (temp 1.0 / top-p 0.95 / top-k 20 /
# min-p 0.0 / repeat-penalty 1.0), and the BF16 mmproj projector so the server
# accepts image and video input. Those are server defaults; a client that sends
# its own temperature still overrides them, which is what the bench does.
#
# repeat-penalty 1.0 is passed explicitly rather than left off: llama.cpp
# defaults it to 1.1, and a default that is merely a different taste is a wrong
# answer for a reasoning model that emits long structured think blocks.
#
# Why master and not a tagged release: llama.cpp merged the qwen35 architecture
# and PROJECTOR_TYPE_QWEN3VL recently enough that the prebuilt win-cuda releases
# still fail the model with "unknown model architecture: 'qwen35'". This script
# updates the same $SRC tree 10-llamacpp-muse.sh uses rather than building a
# second one beside it -- one CUDA build tree, both models, no duplicate 6 GB of
# objects and no ambiguity about which llama-server the config's bin points at.
# Muse Glimmer keeps working across the update; its arch is older than this one.
#
# Why WSL2 and not the Windows host: same as 10-llamacpp-muse.sh. The CUDA
# toolchain is there for vLLM, and the gateway reaches the server over
# localhostForwarding. Run from Windows with:
#   wsl -d Ubuntu bash /mnt/c/Users/gopher/local-llm/scripts/11-llamacpp-qwen38.sh
set -euo pipefail

# HOME is defaulted rather than used bare: this script is long enough to be worth
# running under `systemd-run --unit=... --collect` so it survives the session
# that launched it, and systemd does not set HOME for a Type=oneshot unit. With
# set -u that is an immediate "HOME: unbound variable" abort before anything
# happens, which is a confusing way to lose a 30-minute build.
SRC="${LLAMA_SRC:-${HOME:-/root}/llama.cpp}"
MODELS="${QWEN38_MODELS_DIR:-${HOME:-/root}/models/gguf}"
WIN_STAGE="${WIN_STAGE:-/mnt/c/models/gguf}"
CUDA_ARCH="${CUDA_ARCH:-89}"   # Ada / RTX 4090 is sm_89
PORT="${QWEN38_PORT:-8083}"
REPO="https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main"

# UD-Q4_K_XL is 15.6 GiB and the projector 1.2 GiB. Together with a 32K q8_0 KV
# cache that measures 19.3 GB resident on the 24 GB card -- see bench/README.md
# for why context is close to free on this architecture.
MODEL="Qwen3.8-27B-UD-Q4_K_XL.gguf"
MMPROJ="mmproj-Qwen3.8-27B-BF16.gguf"

export PATH=/usr/local/cuda/bin:$PATH
command -v nvcc >/dev/null || { echo "nvcc not on PATH; is the CUDA toolkit installed in WSL?" >&2; exit 1; }

if [ ! -d "$SRC/.git" ]; then
    echo "cloning llama.cpp into $SRC"
    git clone https://github.com/ggml-org/llama.cpp "$SRC"
fi
cd "$SRC"
git fetch --all --tags --quiet
git checkout master --quiet
git pull --quiet
echo "llama.cpp head: $(git log -1 --format='%h %cI %s')"

# LLAMA_CURL=OFF because the weights are staged below, not pulled by the server.
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build --config Release -j "$(nproc)" \
    --target llama-server llama-cli llama-bench llama-mtmd-cli
"$SRC/build/bin/llama-server" --version

# The arch check is after the build, not before it: unlike muse-glimmer there is
# no single merge commit to test ancestry against, and asking the binary what it
# supports is the honest question anyway.
if ! "$SRC/build/bin/llama-server" --list-devices >/dev/null 2>&1; then
    echo "built llama-server does not run; check the CUDA build above" >&2
    exit 1
fi

# Weights live on the WSL ext4 disk, not /mnt/c: the 9p hop off the Windows
# volume roughly triples cold load time on a 15 GB file.
mkdir -p "$MODELS"
for f in "$MODEL" "$MMPROJ"; do
    if [ -f "$MODELS/$f" ]; then
        echo "have $f"
    elif [ -f "$WIN_STAGE/$f" ]; then
        echo "copying $f from $WIN_STAGE"
        cp "$WIN_STAGE/$f" "$MODELS/$f"
    else
        # -f so an HTTP error fails the run instead of writing the error body to
        # a .gguf, which is how the Muse Glimmer projector went silently missing.
        echo "downloading $f"
        curl -fsSL --retry 5 --retry-all-errors -o "$MODELS/$f" "$REPO/$f"
    fi
done
ls -la "$MODELS"

# A systemd unit, the same way 08-vllm.sh and 10-llamacpp-muse.sh run theirs. A
# backgrounded server dies with the session that launched it, and the gateway
# then fails the request rather than adopting anything (it runs on the Windows
# host and cannot exec this binary). --host 0.0.0.0 so the gateway reaches it
# over localhostForwarding.
#
# Port 8083 rather than Muse Glimmer's 8080: both units can be installed at once,
# but only one should be running, because either model alone is most of the card.
# The gateway's vram_budget_mb is what enforces that.
cat > /etc/systemd/system/llama-qwen38.service <<UNIT
[Unit]
Description=llama-server serving Qwen3.8-27B on port $PORT
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Environment=CUDA_VISIBLE_DEVICES=0
Environment="PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
KillMode=control-group
ExecStart=$SRC/build/bin/llama-server \\
    --model $MODELS/$MODEL \\
    --mmproj $MODELS/$MMPROJ \\
    --host 0.0.0.0 \\
    --port $PORT \\
    --n-gpu-layers 999 \\
    --ctx-size 32768 \\
    --flash-attn on \\
    --cache-type-k q8_0 \\
    --cache-type-v q8_0 \\
    --temp 1.0 \\
    --top-p 0.95 \\
    --top-k 20 \\
    --min-p 0.0 \\
    --repeat-penalty 1.0 \\
    --no-webui \\
    --metrics
Restart=on-failure
RestartSec=30
StandardOutput=journal
StandardError=journal
SyslogIdentifier=llama-qwen38

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
echo "wrote /etc/systemd/system/llama-qwen38.service"

cat <<EOF

Done. Start the server the gateway's qwen3.8-27b entry adopts:

  systemctl disable --now llama-muse-glimmer   # either model is most of the card
  systemctl enable  --now llama-qwen38

Reasoning effort is a chat-template argument, not a server flag. Per request:

  "chat_template_kwargs": {"reasoning_effort": "low"}   # or medium, xhigh (default)

Vision requests need max_tokens headroom: the think block is counted against the
same budget, and at the default effort a 300-token cap returns an empty content
string with completion_tokens at the cap. Budget 2048 or more.
EOF
