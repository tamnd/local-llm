//go:build llama && !llamastatic

// This file supplies the CGO linker flags for the dynamic (shared library)
// build. Compiled only when -tags llama is set WITHOUT -tags llamastatic.
// Use: make build-llama
//
// The shared libraries (libllama.so, libggml*.so) must be in the directory
// pointed to by the -L flag, and LD_LIBRARY_PATH must include that directory
// at runtime. See scripts/build-libllama.sh.
package llama

/*
#cgo LDFLAGS: -L${SRCDIR}/../third_party/llama.cpp/build/bin -lllama -lggml -lggml-base -lggml-cpu -lggml-cuda -lstdc++ -lm
*/
import "C"
