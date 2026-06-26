---
title: "Installation"
description: "Install the llmgw binary from a release archive, go install, Homebrew, Scoop, or a container image."
weight: 20
---

`llmgw` is a single binary. The default build is pure Go, so for most setups installation is just dropping one file on the box and running it. Pick whichever channel fits your machine.

## Release archive

Grab a prebuilt archive for Linux, macOS, or Windows from the [GitHub Releases](https://github.com/tamnd/local-llm/releases) page, unpack it, and put `llmgw` on your `PATH`.

```bash
# Linux x86_64, adjust the asset name to the release you want
curl -L -o llmgw.tar.gz \
  https://github.com/tamnd/local-llm/releases/latest/download/llmgw_linux_amd64.tar.gz
tar xzf llmgw.tar.gz
sudo mv llmgw /usr/local/bin/
```

## go install

If you have a Go 1.26 toolchain, build and install straight from source:

```bash
go install github.com/tamnd/local-llm/cmd/llmgw@latest
```

The binary lands in `$(go env GOPATH)/bin`.

## Homebrew

```bash
brew install tamnd/tap/local-llm
```

## Scoop

```bash
scoop bucket add tamnd https://github.com/tamnd/scoop-bucket
scoop install local-llm
```

## Container image

```bash
docker run --rm -p 8888:8888 ghcr.io/tamnd/local-llm
```

Mount your config and expose the admin plane as needed; see the [configuration reference](/reference/configuration/) for the full set of flags.

## The in-process CUDA engine

The pure-Go binary does not include the in-process `inproc` backend. That path embeds llama.cpp via cgo so the gateway runs the decode loop itself, with no subprocess and no HTTP hop, for maximum tokens per second on the dense tier.

**Prebuilt static binary.** The simplest way to get the inproc engine is to download `llmgw-cuda` from the [GitHub release page](https://github.com/tamnd/local-llm/releases). It statically embeds cuBLAS and cudart, so it needs no extra `.so` files at runtime. Drop it on any Linux machine with an NVIDIA driver ≥ 525:

```bash
chmod +x llmgw-cuda
./llmgw-cuda -config llmgw.yaml
```

**Build from source.** If you need a custom llama.cpp build or a different CUDA version, build it on the box:

```bash
# Build static archives (runs cmake with CUDA)
scripts/build-libllama.sh --static

# Link the static binary
make build-llama-static
```

This produces `bin/llmgw-cuda`, statically linking everything except the NVIDIA driver. The `make build-llama` target (no `--static`) links shared `.so` files and requires `LD_LIBRARY_PATH` at runtime; that path is faster for development iteration.

The configuration reference covers wiring an `inproc` model into your config.

## Verify

```bash
llmgw -version
```
