# 03-ollama.ps1: install Ollama 0.30.6 and run it as the OllamaServe scheduled
# task on loopback, with flash attention and q8 KV cache enabled (spec doc 10
# section 6). Loopback-only: the gateway is the single tailnet-facing service.
. "$PSScriptRoot\common.ps1"

$required = "0.30.6"
$ollamaPath = "C:\Users\gopher\AppData\Local\Programs\Ollama\ollama.exe"

if (Test-Path $ollamaPath) {
    $ver = (& $ollamaPath --version 2>&1)
    Log "Ollama found: $ver"
    if ($ver -notmatch [regex]::Escape($required)) {
        Log "Wrong version, reinstalling $required..."
        $uninst = "C:\Users\gopher\AppData\Local\Programs\Ollama\uninstall.exe"
        if (Test-Path $uninst) { & $uninst /S }
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
    Start-Process $installer -ArgumentList "/S" -Wait
    Log "Ollama installed."
}

# Stop any running instance before the task takes over the loopback port.
$proc = Get-Process ollama -ErrorAction SilentlyContinue
if ($proc) { Stop-Process -Name ollama -Force; Start-Sleep 2 }

# Env vars are set inside the task action (not system-wide) so no logoff cycle is
# needed. Single logical line: no backtick continuation in a schtasks command.
$cmd = '/c "set OLLAMA_FLASH_ATTENTION=1 && set OLLAMA_KV_CACHE_TYPE=q8_0 && set OLLAMA_HOST=127.0.0.1:11434 && C:\Users\gopher\AppData\Local\Programs\Ollama\ollama.exe serve >> ' + $script:LogDir + '\ollama.log 2>&1"'
Register-Service "OllamaServe" $cmd
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
