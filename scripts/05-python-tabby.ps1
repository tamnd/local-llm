# 05-python-tabby.ps1: Python 3.11, a venv, PyTorch 2.9.0+cu128, flash-attn,
# triton-windows, ExLlamaV3, and TabbyAPI on loopback port 5000 as the TabbyServe
# scheduled task (spec doc 10 section 8). TabbyAPI is the fastest single-stream
# decode path for models that fit fully in VRAM. Idempotent at each sub-step.
#
# 2026 version audit (spec doc 19 section 4). The RTX 4090 driver is now 610.62,
# which opens CUDA 13.x, but the ExLlamaV3 arm is held at torch 2.9.0+cu128 and
# that pin is a hard ceiling, not a stale number:
#   - ExLlamaV3 needs flash-attn for its attention path, and there is no Windows
#     flash-attn wheel for torch 2.10 or 2.11, so 2.9 is the newest torch that has
#     a matching Windows flash-attn build (flash_attn 2.8.3 cu128/torch2.9).
#   - triton-windows must be the 3.5 line to match torch 2.9. ExLlamaV3's
#     chunked-attention kernel imports triton at prefill; the 3.7 line moved the
#     autotuner API (arg_names) and raises "'function' object has no attribute
#     'arg_names'", so pin < 3.6.
#   - torchaudio must be removed. The cu128 index resolves a mismatched torchaudio
#     whose _torchaudio.pyd fails to load against torch 2.9, and a recent
#     transformers imports it eagerly through transformers.loss.loss_rnnt, which
#     crashes TabbyAPI at startup. ExLlamaV3 inference does not use torchaudio.
# The driver is forward compatible, so cu128 wheels run fine on the 610.62 / CUDA
# 13 driver; do not chase cu130 wheels here, they force torch 2.11 and lose
# flash-attn on Windows.
. "$PSScriptRoot\common.ps1"

# --- Python 3.11 ---
$pyexe = "C:\Python311\python.exe"
if (Test-Path $pyexe) {
    Log "Python already installed: $(& $pyexe --version)"
} else {
    $installer = "$env:TEMP\python-3.11.9-amd64.exe"
    Log "Downloading Python 3.11.9..."
    Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe" -OutFile $installer
    Log "Installing Python 3.11.9 to C:\Python311..."
    Start-Process $installer -ArgumentList "/quiet InstallAllUsers=0 TargetDir=C:\Python311 PrependPath=0 Include_test=0" -Wait
}
if ((& $pyexe --version) -notmatch "3\.11") { Fail "Expected Python 3.11 at $pyexe." }

# --- venv ---
$venv = Join-Path $RepoRoot "tabby-venv"
if (-not (Test-Path "$venv\Scripts\python.exe")) {
    Log "Creating venv at $venv..."
    & $pyexe -m venv $venv
}
$pip = "$venv\Scripts\pip.exe"

# --- PyTorch 2.9.0 + CUDA 12.8 ---
if ((& $pip show torch 2>&1) -match "Version: 2\.9\.0") {
    Log "PyTorch 2.9.0+cu128 already installed."
} else {
    Log "Installing PyTorch 2.9.0+cu128..."
    & $pip install torch==2.9.0 torchvision --index-url https://download.pytorch.org/whl/cu128
}

# --- Remove torchaudio ---
# The cu128 index resolves a torchaudio whose native _torchaudio.pyd will not load
# against torch 2.9, and a recent transformers imports torchaudio eagerly at
# startup (loss_rnnt), which crashes TabbyAPI. ExLlamaV3 inference does not need
# it, so uninstall it if a dependency dragged it in.
if ((& $pip show torchaudio 2>&1) -match "Name: torchaudio") {
    Log "Removing torchaudio (breaks TabbyAPI startup on torch 2.9)..."
    & $pip uninstall -y torchaudio
}

# --- flash-attn 2.8.3 (cu128 / torch2.9 Windows wheel) ---
# ExLlamaV3's attention path requires flash-attn. There is no Windows flash-attn
# wheel for torch 2.10 or 2.11, which is why torch is held at 2.9. Set
# FLASH_ATTN_WHEEL to the cp311-win_amd64 cu128torch2.9 flash_attn 2.8.3 wheel;
# there is no PyPI Windows fallback, so a missing wheel is a hard stop.
if ((& $pip show flash-attn 2>&1) -match "Version: 2\.8\.3") {
    Log "flash-attn 2.8.3 already installed."
} elseif ($env:FLASH_ATTN_WHEEL) {
    Log "Installing flash-attn from $env:FLASH_ATTN_WHEEL..."
    & $pip install --no-deps $env:FLASH_ATTN_WHEEL
} else {
    Fail "FLASH_ATTN_WHEEL not set. ExLlamaV3 needs flash-attn and there is no Windows PyPI wheel; point it at the cu128torch2.9 cp311 flash_attn 2.8.3 wheel."
}

# --- triton-windows (3.5 line, matches torch 2.9) ---
# ExLlamaV3's chunked-attention kernel imports triton at prefill. Pin < 3.6: the
# 3.7 line changed the autotuner API and breaks the kernel.
if ((& $pip show triton-windows 2>&1) -match "Version: 3\.5") {
    Log "triton-windows 3.5 already installed."
} else {
    Log "Installing triton-windows (3.5 line)..."
    & $pip install "triton-windows<3.6"
}

# --- ExLlamaV3 1.1.0 (prebuilt wheel; the sdist needs Ninja + MSVC) ---
# The 1.1.0 cp311 wheel must be the cu128.torch2.9.0 build, matching the torch pin
# above. Set EXLLAMAV3_WHEEL to that cp311-win_amd64 wheel from the ExLlamaV3 1.1.0
# release; --no-deps so it does not pull a mismatched torch back in.
if ((& $pip show exllamav3 2>&1) -match "Version: 1\.1\.0") {
    Log "ExLlamaV3 1.1.0 already installed."
} else {
    if ($env:EXLLAMAV3_WHEEL) {
        Log "Installing ExLlamaV3 1.1.0 from $env:EXLLAMAV3_WHEEL..."
        & $pip install --no-deps $env:EXLLAMAV3_WHEEL
    } else {
        Log "EXLLAMAV3_WHEEL not set; attempting PyPI install (may build from source)."
        & $pip install exllamav3==1.1.0
    }
}

# --- TabbyAPI + huggingface_hub ---
if ((& $pip show tabbyapi 2>&1) -match "Name: tabbyapi") {
    Log "TabbyAPI already installed."
} else {
    Log "Installing TabbyAPI..."
    & $pip install tabbyapi
}
& $pip install huggingface_hub --quiet
Log "huggingface_hub installed."

# --- TabbyServe task ---
# ExLlamaV3 requires flash-attn, so do NOT set EXLLAMA_NO_FLASH_ATTN here: the
# flash_attn 2.8.3 cu128/torch2.9 wheel installed above is the whole reason torch
# is held at 2.9. Loopback only.
$vpython = "$venv\Scripts\python.exe"
$cmd = '/c "' + $vpython + ' -m tabbyapi.main --host 127.0.0.1 --port 5000 >> ' + $script:LogDir + '\tabby.log 2>&1"'
Register-Service "TabbyServe" $cmd
Start-ScheduledTask -TaskName "TabbyServe"

$ok = $false
foreach ($i in 1..30) {
    if (Test-HttpOk "http://127.0.0.1:5000/v1/models") { $ok = $true; break }
    Start-Sleep 1
}
if ($ok) {
    Log "TabbyServe is up at 127.0.0.1:5000."
} else {
    Fail "TabbyAPI did not answer within 30s. Check $script:LogDir\tabby.log."
}
