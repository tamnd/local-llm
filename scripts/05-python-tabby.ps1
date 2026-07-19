# 05-python-tabby.ps1: Python 3.11, a venv, PyTorch 2.10.0+cu128, ExLlamaV3, and
# TabbyAPI on loopback port 5000 as the TabbyServe scheduled task (spec doc 10
# section 8). TabbyAPI is the fastest single-stream decode path for models that
# fit fully in VRAM. Idempotent at each sub-step.
#
# 2026 version audit (spec doc 19 section 4): the RTX 4090 driver is now 610.62,
# which opens CUDA 13.x. ExLlamaV3 1.1.0 ships no cu124/torch2.6 wheel, so the
# stack is pinned to the latest CUDA-12.8 pair the current wheels support:
# torch 2.10.0+cu128 and the exllamav3 1.1.0 cp311 wheel built against it.
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

# --- PyTorch 2.10.0 + CUDA 12.8 ---
if ((& $pip show torch 2>&1) -match "Version: 2\.10\.0") {
    Log "PyTorch 2.10.0+cu128 already installed."
} else {
    Log "Installing PyTorch 2.10.0+cu128..."
    & $pip install torch==2.10.0+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
}

# --- ExLlamaV3 1.1.0 (prebuilt wheel; the sdist needs Ninja + MSVC) ---
# The 1.1.0 cp311 wheel is built against cu128/torch2.10, matching the torch pin
# above. Set EXLLAMAV3_WHEEL to the cp311-win_amd64 cu128.torch2.10 wheel from the
# ExLlamaV3 1.1.0 release; otherwise install from PyPI as a fallback.
if ((& $pip show exllamav3 2>&1) -match "Version: 1\.1\.0") {
    Log "ExLlamaV3 1.1.0 already installed."
} else {
    if ($env:EXLLAMAV3_WHEEL) {
        Log "Installing ExLlamaV3 1.1.0 from $env:EXLLAMAV3_WHEEL..."
        & $pip install $env:EXLLAMAV3_WHEEL
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
# EXLLAMA_NO_FLASH_ATTN=1: no prebuilt flash-attn wheel for cu124 on Windows as of
# June 2026, so TabbyAPI must not try to import it. Loopback only.
$vpython = "$venv\Scripts\python.exe"
$cmd = '/c "set EXLLAMA_NO_FLASH_ATTN=1 && ' + $vpython + ' -m tabbyapi.main --host 127.0.0.1 --port 5000 >> ' + $script:LogDir + '\tabby.log 2>&1"'
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
