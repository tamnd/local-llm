# 02-openssh.ps1: install and start OpenSSH Server, key-only access over the
# tailnet. Full rationale and the authorized_keys setup are in spec doc 09; this
# script records the enable + autostart steps. Idempotent.
. "$PSScriptRoot\common.ps1"

$sshd = Get-WindowsCapability -Online -Name OpenSSH.Server* | Select-Object -First 1
if ($sshd.State -eq "Installed") {
    Log "OpenSSH.Server already installed."
} else {
    Log "Installing OpenSSH.Server..."
    Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0 | Out-Null
    Log "OpenSSH.Server installed."
}

$service = Get-Service -Name sshd -ErrorAction SilentlyContinue
if ($service -and $service.StartType -eq "Automatic" -and $service.Status -eq "Running") {
    Log "sshd already running, start type Automatic. Done."
    exit 0
}

Set-Service -Name sshd -StartupType Automatic
Start-Service sshd
Log "sshd started and set to Automatic."

if (netstat -an | Select-String ":22 ") {
    Log "Port 22 is listening. SSH is up."
} else {
    Fail "Port 22 is not listening after service start."
}
