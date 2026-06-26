//go:build llama && llamastatic

// This file supplies the CGO linker flags for the static single-binary build.
// Compiled only when BOTH -tags llama AND -tags llamastatic are set.
// Use: make build-llama-static (which also sets CGO_LDFLAGS for CUDA)
//
// The llama.cpp static archive (libllama-full.a) is produced by:
//   scripts/build-libllama.sh --static
// It combines libllama.a, libggml.a, libggml-base.a, libggml-cpu.a, and
// libggml-cuda.a into one archive so the linker only needs a single -l flag.
//
// CUDA runtime libraries are NOT listed here because their paths vary by
// machine. The Makefile passes them via CGO_LDFLAGS:
//   -L${CUDA_ROOT}/lib64 -L${CUDA_ROOT}/lib64/stubs -lcuda -lcudart_static -lculibos
//
// The resulting binary links libcuda.so dynamically (NVIDIA driver, always
// present on any CUDA machine) and statically embeds cudart. It runs on any
// Linux box with an NVIDIA driver >= 525 and no extra .so setup.
package llama

/*
// All linker flags for the static build come from CGO_LDFLAGS in the
// Makefile's build-llama-static target. The Go cgo security rules block
// -Wl flags (commas rejected) and archive paths from #cgo directives.
// This file exists only to carry the build tag.
*/
import "C"
