# 05-python-tabby.ps1: Python 3.11, a venv, PyTorch 2.6.0+cu124, ExLlamaV3, and
# TabbyAPI on loopback port 5000 as the TabbyServe scheduled task (spec doc 10
# section 8). TabbyAPI is the fastest single-stream decode path for models that
# fit fully in VRAM. Idempotent at each sub-step.
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

# --- PyTorch 2.6.0 + CUDA 12.4 ---
if ((& $pip show torch 2>&1) -match "Version: 2\.6\.0") {
    Log "PyTorch 2.6.0+cu124 already installed."
} else {
    Log "Installing PyTorch 2.6.0+cu124..."
    & $pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
}

# --- ExLlamaV3 (prebuilt wheel; the sdist needs Ninja + MSVC) ---
# Pin the exact cp311-win_amd64 cu124 wheel from the ExLlamaV3 releases page.
# Set EXLLAMAV3_WHEEL to override; otherwise install from PyPI as a fallback.
if ((& $pip show exllamav3 2>&1) -match "Name: exllamav3") {
    Log "ExLlamaV3 already installed."
} else {
    if ($env:EXLLAMAV3_WHEEL) {
        Log "Installing ExLlamaV3 from $env:EXLLAMAV3_WHEEL..."
        & $pip install $env:EXLLAMAV3_WHEEL
    } else {
        Log "EXLLAMAV3_WHEEL not set; attempting PyPI install (may build from source)."
        & $pip install exllamav3
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
