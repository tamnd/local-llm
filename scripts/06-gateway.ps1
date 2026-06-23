# 06-gateway.ps1: build llmgw.exe from this repo and run it as the GatewayServe
# scheduled task. The gateway is the only tailnet-facing service; it fronts
# Ollama (11434), TabbyAPI (5000), and any llama-server processes, exposing one
# OpenAI-compatible surface on the data plane (8888) plus a loopback admin plane
# (8889). Idempotent: a present binary and live /healthz short-circuit the build.
. "$PSScriptRoot\common.ps1"

$bin = Join-Path $RepoRoot "bin\llmgw.exe"
$cfg = Join-Path $RepoRoot "configs\llmgw.yaml"

# --- Go toolchain ---
$go = Get-Command go -ErrorAction SilentlyContinue
if (-not $go) { Fail "go not found on PATH. Install Go 1.26+ and re-run." }
Log "Go toolchain: $(& go version)"

# --- build ---
# Stamp the version from git so /healthz and the system_fingerprint match the
# checked-out commit. A detached or shallow clone still yields something useful.
Push-Location $RepoRoot
try {
    $version = (& git describe --tags --always --dirty 2>$null)
    if (-not $version) { $version = "dev" }
    Log "Building llmgw $version..."
    $env:CGO_ENABLED = "0"
    & go build -ldflags "-s -w -X main.version=$version" -o $bin ".\cmd\llmgw"
    if ($LASTEXITCODE -ne 0) { Fail "go build failed." }
} finally {
    Pop-Location
}
if (-not (Test-Path $bin)) { Fail "Build reported success but $bin is missing." }
Log "Built $bin ($(& $bin -version))."

# --- config ---
# The task points at configs/llmgw.yaml. The committed file ships placeholder
# tokens (REPLACE_ME...); refuse to start a tailnet-facing service with those in
# place so a real deploy cannot leak an open gateway.
if (-not (Test-Path $cfg)) { Fail "Missing $cfg. Copy and edit configs/llmgw.yaml before provisioning." }
if (Select-String -Path $cfg -Pattern "REPLACE_ME" -Quiet) {
    Fail "configs/llmgw.yaml still has REPLACE_ME tokens. Set real auth tokens before exposing the gateway."
}

# --- GatewayServe task ---
$cmd = '/c "' + $bin + ' -config ' + $cfg + ' >> ' + $script:LogDir + '\gateway.log 2>&1"'
Register-Service "GatewayServe" $cmd
Start-ScheduledTask -TaskName "GatewayServe"

# /healthz needs no token (see gateway.go: handleInference gates auth, health does
# not). Poll the data plane; the admin plane is loopback and comes up with it.
$ok = $false
foreach ($i in 1..20) {
    if (Test-HttpOk "http://127.0.0.1:8888/healthz") { $ok = $true; break }
    Start-Sleep 1
}
if ($ok) {
    Log "GatewayServe is up at 0.0.0.0:8888 (admin on 127.0.0.1:8889)."
} else {
    Fail "Gateway /healthz did not answer within 20s. Check $script:LogDir\gateway.log."
}
