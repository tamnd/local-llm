# 03-ollama.ps1: install Ollama 0.32.9 and run it as the OllamaServe scheduled
# task on loopback, with flash attention and q8 KV cache enabled (spec doc 10
# section 6). Loopback-only: the gateway is the single tailnet-facing service.
#
# 0.32.7 was the first release to carry Muse Glimmer, but only through the MLX
# engine on Apple Silicon. 0.32.9 is pinned here so the rest of the roster runs
# on a current runtime; whether its registry now serves the GGUF tags to a CUDA
# host is re-checked on each bump, and until it does Muse Glimmer is served by
# llama-server (10-llamacpp-muse.sh).
. "$PSScriptRoot\common.ps1"

# OllamaSetup.exe is an Inno Setup installer, not NSIS. It accepts /VERYSILENT
# and ignores /S, and an ignored switch means the wizard waits for a UI: run
# unattended over SSH it hangs forever in session 0 rather than failing, which
# is exactly what happened on the 0.32.7 -> 0.32.9 bump.
$script:InnoSilent = @("/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/NOCANCEL")

$required = "0.32.9"
$ollamaPath = "C:\Users\gopher\AppData\Local\Programs\Ollama\ollama.exe"

if (Test-Path $ollamaPath) {
    $ver = (& $ollamaPath --version 2>&1)
    Log "Ollama found: $ver"
    if ($ver -notmatch [regex]::Escape($required)) {
        Log "Wrong version, reinstalling $required..."
        $uninst = "C:\Users\gopher\AppData\Local\Programs\Ollama\unins000.exe"
        if (Test-Path $uninst) { Start-Process $uninst -ArgumentList $script:InnoSilent -Wait }
        Start-Sleep 5
    } else {
        Log "Ollama $required already installed."
    }
}

if (-not (Test-Path $ollamaPath)) {
    $installer = "$env:TEMP\OllamaSetup.exe"
    Log "Downloading Ollama $required..."
    Invoke-WebRequest -Uri "https://github.com/ollama/ollama/releases/download/v$required/OllamaSetup.exe" -OutFile $installer
    Log "Installing Ollama $required silently..."
    Start-Process $installer -ArgumentList $script:InnoSilent -Wait
    Log "Ollama installed."
}

# Stop any running instance before the task takes over the loopback port.
$proc = Get-Process ollama -ErrorAction SilentlyContinue
if ($proc) { Stop-Process -Name ollama -Force; Start-Sleep 2 }

# Env vars are set inside the task action (not system-wide) so no logoff cycle is
# needed. Single logical line: no backtick continuation in a schtasks command.
#
# OLLAMA_MODELS is set explicitly because the task runs as SYSTEM (-AsSystem, so
# it starts at boot without a logon). Ollama defaults the store to
# %USERPROFILE%\.ollama\models, which under SYSTEM resolves to a directory in
# config\systemprofile, and the whole 13-model roster would silently look
# unpulled. Session 0 is fine for CUDA: the runner still enumerates the 4090.
$cmd = '/c "set OLLAMA_FLASH_ATTENTION=1 && set OLLAMA_KV_CACHE_TYPE=q8_0 && set OLLAMA_HOST=127.0.0.1:11434 && set OLLAMA_MODELS=C:\Users\gopher\.ollama\models && C:\Users\gopher\AppData\Local\Programs\Ollama\ollama.exe serve >> ' + $script:LogDir + '\ollama.log 2>&1"'
Register-Service "OllamaServe" $cmd -AsSystem
Start-ScheduledTask -TaskName "OllamaServe"

# Wait for the API to answer instead of a fixed sleep.
$ok = $false
foreach ($i in 1..20) {
    if (Test-HttpOk "http://127.0.0.1:11434/api/version") { $ok = $true; break }
    Start-Sleep 1
}
if ($ok) {
    Log "OllamaServe is up at 127.0.0.1:11434."
} else {
    Fail "Ollama API did not answer within 20s. Check $script:LogDir\ollama.log."
}
