# 01-driver.ps1: confirm the NVIDIA driver is at the required version. The driver
# is a manual install (NVIDIA's installer is interactive); this script verifies
# the outcome and tells you exactly what to download if it is wrong. Idempotent:
# a correct driver prints "already at required version" and exits 0.
. "$PSScriptRoot\common.ps1"

$required = "566.36"

$smi = & "C:\Windows\System32\nvidia-smi.exe" --query-gpu=driver_version,compute_cap --format=csv,noheader 2>&1
if ($LASTEXITCODE -ne 0) {
    Fail "nvidia-smi not found. Install the NVIDIA driver first."
}
Log "nvidia-smi: $smi"

$driverVersion = ($smi -split ",")[0].Trim()
if ($driverVersion -eq $required) {
    Log "Driver $driverVersion already at required version. Skipping."
    exit 0
}

Log "Driver is $driverVersion, required $required."
Log "Download Game Ready Driver $required from https://www.nvidia.com/Download/index.aspx"
Log "Select: GeForce RTX 40 Series, RTX 4090, Windows 11 64-bit. Use Custom + clean install, then reboot and re-run."
exit 1
