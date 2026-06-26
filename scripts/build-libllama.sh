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

# Pin to a commit or tag. Bump deliberately and re-measure; see doc 17 on the
# CUDA 13.2 gibberish regression that makes the toolchain version matter.
# b9811 (2026-06-26) is 31 builds past b9780. It includes batched MoE expert
# dispatch and fused SSM kernels for Ada (sm_89), which should close the
# throughput gap vs Ollama on qwen3.6:27b and qwen3.6:35b.
LLAMA_CPP_REF="${LLAMA_CPP_REF:-9df06805eee8d600ccc3cb1b099658c9a91b0bae}"
CUDA_ARCH="${CUDA_ARCH:-89}" # Ada / RTX 4090 is sm_89
# Pin the CUDA toolkit to 12.x when the system has multiple versions installed.
# CUDA 13.x requires a driver >= 590; current GamingPC driver is 566.36 (max 12.7).
CUDA_TOOLKIT_ROOT="${CUDA_TOOLKIT_ROOT:-/usr/local/cuda-12}"

mkdir -p "$THIRD_PARTY"

if [ ! -d "$SRC/.git" ]; then
	echo "cloning llama.cpp @ $LLAMA_CPP_REF"
	git clone https://github.com/ggml-org/llama.cpp "$SRC"
fi

git -C "$SRC" fetch --tags --quiet
git -C "$SRC" checkout --quiet "$LLAMA_CPP_REF"

# Apply compatibility patches for Ollama-format GGUFs (arch name aliases,
# tensor name differences, partial array fills). The patch was written against
# b9780 and none of its hunks are upstream as of b9811, but context lines
# drift across builds. Try strict first, then relaxed (-C1, -C0).
PATCH="$ROOT/patches/llama-b9780-ollama-compat.patch"
if [ -f "$PATCH" ]; then
	applied=0
	for ctx in "" "-C1" "-C0"; do
		if git -C "$SRC" apply --check $ctx "$PATCH" 2>/dev/null; then
			git -C "$SRC" apply $ctx "$PATCH"
			echo "applied $PATCH (context=$ctx)"
			applied=1
			break
		fi
	done
	if [ $applied -eq 0 ]; then
		echo "WARNING: $PATCH did not apply at any context level; Ollama GGUF compat patches skipped"
		echo "  Run: git -C $SRC apply --reject $PATCH"
		echo "  to see which hunks failed, then update the patch for the new llama.cpp version."
	fi
fi

echo "configuring (CUDA arch $CUDA_ARCH, toolkit $CUDA_TOOLKIT_ROOT)"
cmake -S "$SRC" -B "$BUILD" \
	-DCMAKE_BUILD_TYPE=Release \
	-DGGML_CUDA=ON \
	-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
	-DCMAKE_CUDA_COMPILER="$CUDA_TOOLKIT_ROOT/bin/nvcc" \
	-DCUDA_TOOLKIT_ROOT_DIR="$CUDA_TOOLKIT_ROOT" \
	-DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler" \
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
