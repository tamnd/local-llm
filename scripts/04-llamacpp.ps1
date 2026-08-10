# 04-llamacpp.ps1: download the prebuilt llama.cpp b10344 CUDA 13.3 build and
# extract it to C:\llama.cpp (spec doc 10 section 7). llama.cpp reads the same
# GGUF blobs Ollama already pulled, so no second download of weights. Idempotent.
#
# CUDA 13.3 rather than 12.4: the box driver is 610.62, well past the 590 floor
# CUDA 13.x needs, and the 13.3 archive is a third the size. Upstream also split
# the CUDA runtime out of the main archive and dropped the "cu" prefix from the
# asset names since b9553, so both zips are fetched by their current names.
#
# This build does NOT carry the muse-glimmer architecture: it landed in master
# at 62bf73d2, after every tag up to b10344 was cut. The Muse Glimmer entries in
# the gateway config are served by the WSL2 master build that
# scripts/10-llamacpp-muse.sh produces. Once a tag includes the arch, bump $tag
# here and that script can retire.
. "$PSScriptRoot\common.ps1"

$tag = "b10344"
$cuda = "13.3"
$dest = "C:\llama.cpp"
$cli = "$dest\llama-cli.exe"

if (Test-Path $cli) {
    Log "llama.cpp already installed at $dest. Skipping."
    exit 0
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
