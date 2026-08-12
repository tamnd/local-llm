#!/usr/bin/env bash
# 10-llamacpp-muse.sh: build llama.cpp from master with CUDA inside WSL2 and
# stage the Muse Glimmer GGUF set, so the gateway's muse-glimmer-30b entry has a
# server to adopt.
#
# The served configuration follows Unsloth's own run page for this model
# (https://unsloth.ai/docs/models/muse-glimmer): the UD-Q4_K_XL dynamic quant,
# temp 1.0 / top-p 0.95 / top-k 64, and the BF16 mmproj projector so the server
# accepts image input. Those sampling values are server defaults; a client that
# sends its own temperature still overrides them, which is what the bench does
# to keep its numbers comparable across runs.
#
# Why master and not a tagged release: llama.cpp merged the muse-glimmer
# architecture in 62bf73d2 (PR #26841). The prebuilt win-cuda release binaries
# lag master by several hours, and every tag up to b10344 still fails the model
# with "unknown model architecture: 'muse-glimmer'". Once a tagged build carries
# the arch, 04-llamacpp.ps1 can serve this on the Windows side instead.
#
# Why WSL2 and not the Windows host: the CUDA toolchain is already there for
# vLLM (08-vllm.sh), and the gateway reaches the server over
# localhostForwarding, the same path VLLM.Load uses. Run from Windows with:
#   wsl -d Ubuntu bash /mnt/c/Users/gopher/local-llm/scripts/10-llamacpp-muse.sh
set -euo pipefail

SRC="${LLAMA_SRC:-$HOME/llama.cpp}"
MODELS="${MUSE_MODELS_DIR:-$HOME/models/gguf}"
WIN_STAGE="${WIN_STAGE:-/mnt/c/models/gguf}"
CUDA_ARCH="${CUDA_ARCH:-89}"   # Ada / RTX 4090 is sm_89
REPO="https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF/resolve/main"

# The weights: the dynamic 4-bit quant Unsloth recommends as the starting point
# (https://unsloth.ai/docs/models/muse-glimmer), Meta's DFlash draft head for
# speculative decoding, and the vision projector.
#
# MMPROJ is the BF16 projector the Unsloth page names, not the kquant one an
# earlier revision of this script reached for: the repo has no bare
# "mmproj-kquant.gguf" at the path this script builds, so that fetch 404'd and,
# because curl ran without -f, wrote the HTML error body to the target and
# exited 0. The projector was silently absent and the model served text-only.
# Every fetch below now uses -f so a bad name fails the run instead of staging
# a 500-byte "GGUF".
MODEL="Muse-Glimmer-30B-UD-Q4_K_XL.gguf"
DRAFT="dflash-kquant.gguf"
MMPROJ="mmproj-Muse-Glimmer-30B-BF16.gguf"

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
if ! git merge-base --is-ancestor 62bf73d2 HEAD; then
    echo "this checkout predates muse-glimmer support (62bf73d2)" >&2
    exit 1
fi

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

# Weights live on the WSL ext4 disk, not /mnt/c: the 9p hop off the Windows
# volume roughly triples cold load time on a 15 GB file. If a prior Windows-side
# run already staged a blob under C:\models\gguf, copy it across instead of
# re-downloading it.
mkdir -p "$MODELS"
for f in "$MODEL" "$DRAFT" "$MMPROJ"; do
    if [ -f "$MODELS/$f" ]; then
        echo "have $f"
    elif [ -f "$WIN_STAGE/$f" ]; then
        echo "copying $f from $WIN_STAGE"
        cp "$WIN_STAGE/$f" "$MODELS/$f"
    else
        echo "downloading $f"
        curl -fsSL --retry 3 -o "$MODELS/$f" "$REPO/$f"
    fi
done
ls -la "$MODELS"

# A systemd unit, the same way 08-vllm.sh runs the vLLM arms. The distro boots
# systemd as PID 1, so a unit is what survives: a backgrounded server dies with
# the session that launched it, and the gateway then fails the request rather
# than adopting anything (it runs on the Windows host and cannot exec this
# binary). --host 0.0.0.0 so the gateway reaches it over localhostForwarding.
cat > /etc/systemd/system/llama-muse-glimmer.service <<UNIT
[Unit]
Description=llama-server serving Muse Glimmer 30B on port 8080
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
    --port 8080 \\
    --n-gpu-layers 999 \\
    --ctx-size 32768 \\
    --flash-attn on \\
    --cache-type-k q8_0 \\
    --cache-type-v q8_0 \\
    --temp 1.0 \\
    --top-p 0.95 \\
    --top-k 64 \\
    --no-webui \\
    --metrics
Restart=on-failure
RestartSec=30
StandardOutput=journal
StandardError=journal
SyslogIdentifier=llama-muse-glimmer

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
echo "wrote /etc/systemd/system/llama-muse-glimmer.service"

cat <<EOF

Done. Start the server the gateway's muse-glimmer-30b entry adopts:

  systemctl enable --now llama-muse-glimmer

Windows does not start WSL at boot, and wsl.exe needs a logon session, so a
scheduled task under S4U cannot bring the distro up. Register the boot task
with Interactive logon and an At-LogOn trigger, as OllamaServe already is, or
on a headless box start the distro from the first SSH session:

  wsl -d Ubuntu -u root -e /bin/systemctl is-system-running --wait

For the DFlash arm (currently a no-op, see configs/llmgw.yaml), append:

  --spec-draft-model $MODELS/$DRAFT --spec-draft-ngl 99 --spec-draft-n-max 6 --spec-draft-n-min 1
EOF
