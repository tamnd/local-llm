# 04-llamacpp.ps1: download the prebuilt llama.cpp b10362 CUDA 13.3 build and
# extract it to C:\llama.cpp (spec doc 10 section 7). llama.cpp reads the same
# GGUF blobs Ollama already pulled, so no second download of weights. Idempotent,
# and re-extracts when $tag moves so a bump here actually upgrades the box.
#
# CUDA 13.3 rather than 12.4: the box driver is 610.62, well past the 590 floor
# CUDA 13.x needs, and the 13.3 archive is a third the size. Upstream also split
# the CUDA runtime out of the main archive and dropped the "cu" prefix from the
# asset names since b9553, so both zips are fetched by their current names.
#
# b10362 is the first tagged win-cuda build that carries the muse-glimmer
# architecture: it landed in master at 62bf73d2, after b10344 was cut. Verified
# on the box rather than assumed -- the arch name is a compiled-in string, and
# llama.dll, llama-common.dll and mtmd.dll all contain it, so the vision path is
# in this build too. It ships llama-server.exe and llama-mtmd-cli.exe.
#
# That means the WSL2 build in scripts/10-llamacpp-muse.sh is no longer the only
# way to serve Muse Glimmer, and retiring it would also remove the reasons the
# gateway can only adopt that server rather than spawn it, and the At-LogOn task
# that has to start the distro. Not done here: the weights live on the WSL ext4
# disk and are not reachable from the Windows side (\\wsl.localhost fails), so
# the move needs the GGUFs restaged and the whole sweep re-measured on the
# Windows build before the config points at it. All numbers in bench/README.md
# are from the WSL build.
. "$PSScriptRoot\common.ps1"

$tag = "b10362"
$cuda = "13.3"
$dest = "C:\llama.cpp"
$cli = "$dest\llama-cli.exe"

# llama-cli reports the build as a bare number ("version: 10362 (2b1cd1e5)"),
# so the pinned tag is compared with its leading "b" stripped. It writes that to
# stderr, which PowerShell would turn into a terminating NativeCommandError under
# this script's ErrorActionPreference, hence the redirect through cmd.
if (Test-Path $cli) {
    $want = $tag.TrimStart("b")
    $out = cmd /c "`"$cli`" --version 2>&1"
    $m = [regex]::Match(($out -join "`n"), "version:\s*(\d+)")
    $have = if ($m.Success) { $m.Groups[1].Value } else { "unknown" }
    if ($have -eq $want) {
        Log "llama.cpp $tag already installed at $dest. Skipping."
        exit 0
    }
    Log "llama.cpp build $have installed, want $want. Upgrading."
}

New-Item -ItemType Directory -Force -Path $dest | Out-Null
foreach ($name in @("llama-$tag-bin-win-cuda-$cuda-x64.zip", "cudart-llama-bin-win-cuda-$cuda-x64.zip")) {
    $zip = "$env:TEMP\$name"
    Log "Downloading $name..."
    Invoke-WebRequest -Uri "https://github.com/ggml-org/llama.cpp/releases/download/$tag/$name" -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath $dest -Force
}
Log "Extracted to $dest."

if (Test-Path $cli) {
    Log "llama-cli.exe present. Done."
} else {
    Get-ChildItem $dest | ForEach-Object { Log "  $($_.Name)" }
    Fail "llama-cli.exe not found after extraction. Wrong zip? Use the cuda-cu12.4 build, not avx."
}
