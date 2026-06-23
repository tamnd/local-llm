# ollama-bench.ps1: measure decode throughput, prompt processing, cold-load time,
# and resident VRAM for each model served by the local Ollama daemon. It uses
# Ollama's own server-side timing fields (load_duration, prompt_eval_duration,
# eval_duration and the matching counts), so the numbers are measured at the
# runtime, not over the network, and the tailnet RTT does not enter the result.
#
# For each model: evict everything, cold-load it (record load_duration), sample
# resident VRAM, run N measured generations at a fixed prompt and output length,
# average the decode rate, then evict it again. Results print as a table and land
# in bench/results-ollama.json next to this script.
param(
    [string]$OllamaUrl = "http://127.0.0.1:11434",
    [int]$NumPredict = 256,
    [int]$Measured = 3,
    # Comma-separated so it survives the cmd -> powershell -File boundary as one
    # token; split into a real list below. Empty means every model the daemon has.
    [string]$ModelList = ""
)

$ErrorActionPreference = "Stop"
$ollamaExe = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
$outFile = Join-Path $PSScriptRoot "results-ollama.json"

# A fixed prompt of a few hundred tokens so prompt-processing numbers are
# comparable across models. The actual token count is read back from
# prompt_eval_count per model (tokenizers differ), so this is a stable input, not
# an exact token target.
$prompt = (@"
You are a careful systems engineer. Explain, in detail and step by step, how a
write-ahead log guarantees durability and crash recovery in a database engine.
Cover the log record format, the checkpoint mechanism, the redo and undo phases
of recovery, group commit, and the fsync ordering that makes the whole scheme
correct. Then describe how this interacts with page caching and how a buffer pool
decides which dirty pages to flush. Be concrete and use examples where it helps.
"@) * 2

function Get-VramUsedMB {
    $v = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
    return [int]($v | Select-Object -First 1).Trim()
}

function Invoke-Generate($model, $keepAlive) {
    $body = @{
        model      = $model
        prompt     = $prompt
        stream     = $false
        keep_alive = $keepAlive
        options    = @{ num_predict = $NumPredict; temperature = 0 }
    } | ConvertTo-Json -Depth 6
    return Invoke-RestMethod -Uri "$OllamaUrl/api/generate" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 600
}

function Evict($model) {
    $body = @{ model = $model; prompt = ""; stream = $false; keep_alive = 0 } | ConvertTo-Json
    try { Invoke-RestMethod -Uri "$OllamaUrl/api/generate" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 60 | Out-Null } catch {}
    Start-Sleep -Seconds 2
}

if ($ModelList.Trim()) {
    $Models = $ModelList.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
} else {
    # Default to every model the daemon knows about.
    $Models = (& $ollamaExe list) | Select-Object -Skip 1 | ForEach-Object { ($_ -split "\s+")[0] } | Where-Object { $_ }
}

# Evict everything in the set first so the idle baseline is a clean GPU, not one
# with a model left resident from a previous run.
foreach ($m in $Models) { Evict $m }
Start-Sleep -Seconds 2
$idleVram = Get-VramUsedMB
Write-Host "Idle VRAM: $idleVram MiB"
Write-Host "Models: $($Models -join ', ')"
Write-Host ""

$results = @()
foreach ($model in $Models) {
    Write-Host "=== $model ===" -ForegroundColor Cyan
    Evict $model

    # Cold load: the first generation after an evict pays the full load cost,
    # reported by Ollama as load_duration (nanoseconds).
    $cold = Invoke-Generate $model "5m"
    $loadMs = [math]::Round($cold.load_duration / 1e6, 0)
    Start-Sleep -Seconds 1
    $residentVram = Get-VramUsedMB
    $modelVram = $residentVram - $idleVram

    # Measured runs: the model is warm, so load_duration is ~0 and the decode and
    # prompt-eval rates are the steady-state numbers.
    $decodeRates = @()
    $promptRates = @()
    $ttftMs = @()
    $promptTokens = 0
    $evalTokens = 0
    for ($i = 0; $i -lt $Measured; $i++) {
        $r = Invoke-Generate $model "5m"
        if ($r.eval_duration -gt 0) { $decodeRates += $r.eval_count / ($r.eval_duration / 1e9) }
        if ($r.prompt_eval_duration -gt 0) { $promptRates += $r.prompt_eval_count / ($r.prompt_eval_duration / 1e9) }
        # Warm TTFT: model already resident, so first token arrives after the
        # prompt is processed. load_duration is ~0 on a warm run.
        $ttftMs += [math]::Round(($r.load_duration + $r.prompt_eval_duration) / 1e6, 0)
        $promptTokens = $r.prompt_eval_count
        $evalTokens = $r.eval_count
    }

    $decodeAvg = if ($decodeRates.Count) { [math]::Round(($decodeRates | Measure-Object -Average).Average, 1) } else { 0 }
    $promptAvg = if ($promptRates.Count) { [math]::Round(($promptRates | Measure-Object -Average).Average, 1) } else { 0 }
    $ttftAvg = if ($ttftMs.Count) { [math]::Round(($ttftMs | Measure-Object -Average).Average, 0) } else { 0 }

    $row = [ordered]@{
        model            = $model
        decode_tok_s     = $decodeAvg
        prompt_tok_s     = $promptAvg
        cold_load_ms     = $loadMs
        warm_ttft_ms     = $ttftAvg
        resident_vram_mb = $modelVram
        prompt_tokens    = $promptTokens
        output_tokens    = $evalTokens
        runs             = $Measured
    }
    $results += [pscustomobject]$row
    Write-Host ("  decode {0} tok/s | prompt {1} tok/s | load {2} ms | ttft {3} ms | vram {4} MiB" -f $decodeAvg, $promptAvg, $loadMs, $ttftAvg, $modelVram)
    Evict $model
}

Write-Host ""
$results | Format-Table -AutoSize
$payload = [ordered]@{
    generated_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
    gpu          = (& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | Select-Object -First 1)
    idle_vram_mb = $idleVram
    num_predict  = $NumPredict
    measured     = $Measured
    results      = $results
}
$payload | ConvertTo-Json -Depth 6 | Set-Content -Path $outFile -Encoding utf8
Write-Host "Wrote $outFile"
