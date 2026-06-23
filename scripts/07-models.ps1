# 07-models.ps1: pull every model the gateway config references, reading the
# source-of-truth list from catalog/models.manifest.yaml. Ollama models are
# pulled by name; TabbyAPI EXL2 quants are downloaded from Hugging Face into the
# Tabby models dir via hf-download.py. Idempotent: ollama pull is a no-op when the
# blob is current, and hf-download.py skips a dir that already holds the revision.
. "$PSScriptRoot\common.ps1"

$manifest = Join-Path $RepoRoot "catalog\models.manifest.yaml"
if (-not (Test-Path $manifest)) { Fail "Missing $manifest." }

# powershell-yaml parses the manifest. It is the one external module the scripts
# need; install it for the current user if absent.
if (-not (Get-Module -ListAvailable -Name powershell-yaml)) {
    Log "Installing powershell-yaml module..."
    Install-Module powershell-yaml -Scope CurrentUser -Force
}
Import-Module powershell-yaml
$doc = ConvertFrom-Yaml (Get-Content -Raw $manifest)

# --- Ollama models ---
$ollama = "C:\Users\gopher\AppData\Local\Programs\Ollama\ollama.exe"
if (-not (Test-Path $ollama)) { Fail "Ollama not installed. Run 03-ollama.ps1 first." }
if (-not (Test-HttpOk "http://127.0.0.1:11434/api/version")) { Fail "OllamaServe is not up." }

foreach ($m in $doc.ollama.models) {
    Log "ollama pull $($m.name)  ($($m.note))"
    & $ollama pull $m.name
    if ($LASTEXITCODE -ne 0) { Fail "ollama pull $($m.name) failed." }
}

# --- TabbyAPI EXL2 models ---
$venvPy = Join-Path $RepoRoot "tabby-venv\Scripts\python.exe"
$hf = Join-Path $PSScriptRoot "hf-download.py"
if ($doc.tabby -and $doc.tabby.models) {
    $modelsDir = $doc.tabby.models_dir
    New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null
    if (-not (Test-Path $venvPy)) { Fail "Tabby venv missing. Run 05-python-tabby.ps1 first." }
    foreach ($m in $doc.tabby.models) {
        $target = Join-Path $modelsDir $m.dir
        $repo = $m.source -replace "^hf:", ""
        Log "hf download $repo@$($m.revision) -> $target"
        & $venvPy $hf --repo $repo --revision $m.revision --dest $target
        if ($LASTEXITCODE -ne 0) { Fail "Download of $repo failed." }
    }
}

# llama and vllm entries are examples not wired into the default gateway config,
# so they are skipped here. Add the pull steps when you wire those backends in.
Log "All manifest models present."
