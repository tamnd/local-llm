# 09-wsl-keepalive.ps1: register a Windows Task Scheduler job that keeps a
# WSL2 session alive permanently after Windows login.
#
# Context: vLLM runs in WSL2 as a systemd service. The RTX 4090 uses Windows
# WDDM, which ties GPU contexts to Windows-side process sessions. When the
# last wsl.exe session from the Windows side exits, WDDM releases the GPU
# context, which kills the EngineCore subprocess (SIGTERM arrives ~15 seconds
# later via systemd's cgroup cleanup). The fix is to keep one wsl.exe process
# alive at all times so WDDM considers the WSL2 GPU context still in use.
#
# This script registers a Task Scheduler job "WSL-Keepalive" that runs
# `wsl.exe -- sleep infinity` at user login. The job is set to run hidden,
# with no UI. Run this script once as Administrator from PowerShell.
#
# Usage:
#   .\09-wsl-keepalive.ps1
#
# To remove:
#   Unregister-ScheduledTask -TaskName "WSL-Keepalive" -Confirm:$false

$taskName = "WSL-Keepalive"
$taskDescription = "Keeps a WSL2 session open so the WDDM GPU context is not released, allowing vLLM systemd services to hold their GPU context across all logins."

$action = New-ScheduledTaskAction `
    -Execute "wsl.exe" `
    -Argument "-- sleep infinity"

$trigger = New-ScheduledTaskTrigger -AtLogOn

$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Seconds 0) `
    -MultipleInstances IgnoreNew `
    -Hidden

$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -RunLevel Highest

# Remove any existing task first (idempotent)
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Host "Removed existing $taskName task."
}

Register-ScheduledTask `
    -TaskName $taskName `
    -Description $taskDescription `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal | Out-Null

Write-Host "Registered Task Scheduler job '$taskName'."
Write-Host "Starting it now (will auto-run on next login)..."

Start-ScheduledTask -TaskName $taskName

Write-Host ""
Write-Host "WSL2 keepalive is active. The GPU context will now persist across"
Write-Host "all Windows user sessions. vLLM systemd services will no longer"
Write-Host "die when the last manual WSL2 session closes."
Write-Host ""
Write-Host "To verify the keepalive is running:"
Write-Host "  Get-Process wsl | Where-Object { \$_.CPU -lt 1 }"
