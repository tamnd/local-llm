//go:build !llama

package llama

// This file is compiled for every build that does not set the "llama" tag, which
// is the default: the macOS dev build, CI, and any box without a CUDA-linked
// libllama. It satisfies the package API without pulling in cgo, so `go build`,
// `go vet`, and `go test` stay green with no C toolchain. The real engine lives
// in cgo.go behind the tag.

// Available reports whether this binary was built with the in-process engine. In
// the stub build it is always false, which the backend adapter checks at startup
// so it can fail a misconfigured inproc model with a clear message instead of a
// nil dereference.
func Available() bool { return false }

// New always fails in the stub build. Rebuild with `-tags llama` and a libllama
// linked for CUDA to get a working Runner.
func New(Params) (Runner, error) { return nil, ErrUnsupported }
