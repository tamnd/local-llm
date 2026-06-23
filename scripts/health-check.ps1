# health-check.ps1: one command that reports pass/fail for every layer of the
# box, from the driver up to an end-to-end completion through the gateway. Run it
# after provisioning and any time the box looks wrong. It never changes state; it
# only probes. Exit code is 0 when every check passes, 1 otherwise, so it doubles
# as a readiness gate in a script.
. "$PSScriptRoot\common.ps1"

$results = @()
function Check($name, [scriptblock]$test) {
    try {
        if (& $test) { $results += [pscustomobject]@{ Name = $name; Ok = $true }; Log "PASS  $name" }
        else { $results += [pscustomobject]@{ Name = $name; Ok = $false }; Log "FAIL  $name" }
    } catch {
        $results += [pscustomobject]@{ Name = $name; Ok = $false }
        Log "FAIL  $name ($($_.Exception.Message))"
    }
}

# Layer 1: GPU driver answers and reports the expected version.
Check "nvidia-driver" {
    $v = & "C:\Windows\System32\nvidia-smi.exe" --query-gpu=driver_version --format=csv,noheader 2>&1
    $LASTEXITCODE -eq 0 -and "$v".Trim() -eq "566.36"
}

# Layer 2: each backend answers on its loopback port.
Check "ollama"  { Test-HttpOk "http://127.0.0.1:11434/api/version" }
Check "tabby"   { Test-HttpOk "http://127.0.0.1:5000/v1/models" }

# Layer 3: gateway health on the data plane and admin status on the loopback
# plane. Admin status needs the admin token; read it from the config so the check
# stays truthful to what is deployed.
Check "gateway-health" { Test-HttpOk "http://127.0.0.1:8888/healthz" }

$cfg = Join-Path $RepoRoot "configs\llmgw.yaml"
$adminToken = $null
$dataToken = $null
if (Test-Path $cfg) {
    Import-Module powershell-yaml -ErrorAction SilentlyContinue
    if (Get-Module powershell-yaml) {
        $c = ConvertFrom-Yaml (Get-Content -Raw $cfg)
        $adminToken = $c.auth.admin_token
        if ($c.auth.tokens) { $dataToken = $c.auth.tokens[0] }
    }
}

Check "gateway-admin-status" {
    if (-not $adminToken) { throw "no admin_token in config" }
    $h = @{ Authorization = "Bearer $adminToken" }
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:8889/admin/status" -Headers $h -UseBasicParsing -TimeoutSec 5
    $r.StatusCode -eq 200
}

# Layer 4: a real completion through the gateway, default model, data-plane token.
# This exercises auth, routing, the VRAM steward, and an upstream backend in one
# shot. It is the check that actually proves the box can serve.
Check "end-to-end-completion" {
    if (-not $dataToken) { throw "no data token in config" }
    $h = @{ Authorization = "Bearer $dataToken"; "Content-Type" = "application/json" }
    $body = @{
        messages = @(@{ role = "user"; content = "Reply with the single word: ok" })
        max_tokens = 8
        stream = $false
    } | ConvertTo-Json -Depth 5
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:8888/v1/chat/completions" -Method Post -Headers $h -Body $body -UseBasicParsing -TimeoutSec 120
    $r.StatusCode -eq 200 -and ($r.Content -match '"choices"')
}

# --- summary ---
$failed = @($results | Where-Object { -not $_.Ok })
Log "----"
Log "$($results.Count - $failed.Count)/$($results.Count) checks passed."
if ($failed.Count -gt 0) {
    Log "Failing: $(($failed | ForEach-Object { $_.Name }) -join ', ')"
    exit 1
}
Log "All systems go."
exit 0
