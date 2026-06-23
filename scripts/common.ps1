# Shared helpers dot-sourced by every provisioning script. The spec (doc 10
# section 3) repeats a Log function in each script; this factors it out. Each
# script dot-sources it with `. "$PSScriptRoot\common.ps1"`, so the whole repo
# must be present on the box (it is: doc 10 section 2 clones it to C:\local-llm).

$ErrorActionPreference = "Stop"

# Repo root is the parent of scripts/. All paths derive from here so a clone to
# any directory works, not just C:\local-llm.
$script:RepoRoot = Split-Path -Parent $PSScriptRoot
$script:LogDir = Join-Path $RepoRoot "logs"
$script:LogFile = Join-Path $LogDir "provision.log"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Log($msg) {
    $ts = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    "$ts  $msg" | Tee-Object -FilePath $script:LogFile -Append
}

# Fail logs an error and exits non-zero. Use for an unrecoverable step so the
# ordered run stops instead of cascading into the next script.
function Fail($msg) {
    Log "ERROR: $msg"
    exit 1
}

# Test-HttpOk returns $true when a GET to url answers 2xx within the timeout. It
# is the building block for the health gates between steps.
function Test-HttpOk($url, $timeoutSec = 5) {
    try {
        $resp = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec $timeoutSec
        return $resp.StatusCode -ge 200 -and $resp.StatusCode -lt 300
    } catch {
        return $false
    }
}

# Register-Service registers (or replaces) an at-startup scheduled task running
# as gopher with highest privilege. cmdLine is the full cmd.exe argument string.
# This is the one place the task shape is defined, so OllamaServe, TabbyServe,
# and GatewayServe stay identical except for their command.
function Register-Service($name, $cmdLine) {
    $existing = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
    if ($existing) {
        Unregister-ScheduledTask -TaskName $name -Confirm:$false
        Log "Removed stale $name task."
    }
    $action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument $cmdLine
    $trigger = New-ScheduledTaskTrigger -AtStartup
    $principal = New-ScheduledTaskPrincipal -UserId "gopher" -LogonType Interactive -RunLevel Highest
    $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit 0 -RestartInterval (New-TimeSpan -Minutes 1) -RestartCount 3
    Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
    Log "$name scheduled task registered."
}
