#!/usr/bin/env bash
# Build libllama with CUDA so the in-process engine (-tags llama) can link it.
#
# This is the recommended runtime path for the high-throughput zero-proxy engine
# on the RTX 4090 box: run it under WSL2 or native Linux, where the cgo plus CUDA
# toolchain is far less painful than native Windows. Ollama stays the zero-config
# Windows-native fallback. Spec 2065 doc 16 covers the toolchain in full.
#
# It clones llama.cpp into third_party/, builds the shared libraries with the CUDA
# backend for the Ada architecture (sm_89), and leaves them where cgo.go's
# LDFLAGS expect them (third_party/llama.cpp/build/bin). After this, build the
# gateway with: make build-llama
#
# Requirements: git, cmake, a C/C++ toolchain, and the CUDA toolkit (nvcc) on
# PATH. Pin LLAMA_CPP_REF to a known-good tag rather than tracking master so the
# engine builds reproducibly.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="$ROOT/third_party"
SRC="$THIRD_PARTY/llama.cpp"
BUILD="$SRC/build"

# Pin to a release tag. Bump deliberately and re-measure; see doc 17 on the
# CUDA 13.2 gibberish regression that makes the toolchain version matter.
LLAMA_CPP_REF="${LLAMA_CPP_REF:-b9780}"
CUDA_ARCH="${CUDA_ARCH:-89}" # Ada / RTX 4090 is sm_89

mkdir -p "$THIRD_PARTY"

if [ ! -d "$SRC/.git" ]; then
	echo "cloning llama.cpp @ $LLAMA_CPP_REF"
	git clone https://github.com/ggml-org/llama.cpp "$SRC"
fi

git -C "$SRC" fetch --tags --quiet
git -C "$SRC" checkout --quiet "$LLAMA_CPP_REF"

echo "configuring (CUDA arch $CUDA_ARCH)"
cmake -S "$SRC" -B "$BUILD" \
	-DCMAKE_BUILD_TYPE=Release \
	-DGGML_CUDA=ON \
	-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
	-DBUILD_SHARED_LIBS=ON \
	-DLLAMA_BUILD_TESTS=OFF \
	-DLLAMA_BUILD_EXAMPLES=OFF \
	-DLLAMA_BUILD_SERVER=OFF

echo "building"
cmake --build "$BUILD" --config Release -j "$(nproc)" --target llama ggml

echo
echo "libllama built under $BUILD/bin"
echo "now run: make build-llama"
echo
echo "at runtime the loader must find the shared objects, e.g."
echo "  export LD_LIBRARY_PATH=\"$BUILD/bin:\$LD_LIBRARY_PATH\""
