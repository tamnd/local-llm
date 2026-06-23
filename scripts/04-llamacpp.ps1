# 04-llamacpp.ps1: download the prebuilt llama.cpp b9553 CUDA 12.4 build and
# extract it to C:\llama.cpp (spec doc 10 section 7). llama.cpp reads the same
# GGUF blobs Ollama already pulled, so no second download of weights. Idempotent.
. "$PSScriptRoot\common.ps1"

$tag = "b9553"
$dest = "C:\llama.cpp"
$cli = "$dest\llama-cli.exe"

if (Test-Path $cli) {
    Log "llama.cpp already installed at $dest. Skipping."
    exit 0
}

New-Item -ItemType Directory -Force -Path $dest | Out-Null
$zip = "$env:TEMP\llama-$tag-bin-win-cuda-cu12.4-x64.zip"
$url = "https://github.com/ggml-org/llama.cpp/releases/download/$tag/llama-$tag-bin-win-cuda-cu12.4-x64.zip"

Log "Downloading llama.cpp $tag (CUDA 12.4)..."
Invoke-WebRequest -Uri $url -OutFile $zip
Expand-Archive -Path $zip -DestinationPath $dest -Force
Log "Extracted to $dest."

if (Test-Path $cli) {
    Log "llama-cli.exe present. Done."
} else {
    Get-ChildItem $dest | ForEach-Object { Log "  $($_.Name)" }
    Fail "llama-cli.exe not found after extraction. Wrong zip? Use the cuda-cu12.4 build, not avx."
}
