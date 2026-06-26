# Developer entry points. CI runs the same steps (see .github/workflows/ci.yml),
# so a green `make check` locally means a green pipeline.

GO       ?= go
BINARY   := llmgw
PKG      := ./...
VERSION  := $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
LDFLAGS  := -X main.version=$(VERSION)

.PHONY: all
all: check build

.PHONY: build
build: ## Build the llmgw binary into ./bin (stub engine, no cgo)
	$(GO) build -ldflags "$(LDFLAGS)" -o bin/$(BINARY) ./cmd/llmgw

.PHONY: build-llama
build-llama: ## Build llmgw with the in-process cgo engine (needs libllama, see scripts/build-libllama.sh)
	CGO_ENABLED=1 $(GO) build -tags llama -ldflags "$(LDFLAGS)" -o bin/$(BINARY) ./cmd/llmgw

# CUDA root for the static single-binary build. Override on machines where CUDA
# lives elsewhere, e.g. CUDA_ROOT=/usr/local/cuda-12.6 make build-llama-static.
CUDA_ROOT ?= /usr/local/cuda-12

LIBLLAMA_FULL := $(abspath third_party/llama.cpp/build/lib/libllama-full.a)

.PHONY: build-llama-static
build-llama-static: ## Build a single static CUDA binary (no .so deps at runtime)
	scripts/build-libllama.sh --static
	CGO_ENABLED=1 \
	CGO_LDFLAGS="-Wl,--whole-archive $(LIBLLAMA_FULL) -Wl,--no-whole-archive -L$(CUDA_ROOT)/lib64 -L$(CUDA_ROOT)/lib64/stubs -lcuda -lcudart_static -lculibos" \
	$(GO) build -tags "llama llamastatic" -ldflags "$(LDFLAGS)" -o bin/$(BINARY)-cuda ./cmd/llmgw

.PHONY: libllama
libllama: ## Compile libllama shared libs with CUDA for the in-process engine
	scripts/build-libllama.sh

.PHONY: test
test: ## Run the test suite with the race detector
	$(GO) test -race -count=1 $(PKG)

.PHONY: vet
vet: ## Run go vet
	$(GO) vet $(PKG)

.PHONY: fmt
fmt: ## Format all Go files
	gofmt -w .

.PHONY: fmt-check
fmt-check: ## Fail if any file is not gofmt-clean
	@unformatted=$$(gofmt -l .); \
	if [ -n "$$unformatted" ]; then echo "needs gofmt:"; echo "$$unformatted"; exit 1; fi

.PHONY: lint
lint: ## Run golangci-lint
	golangci-lint run $(PKG)

.PHONY: check
check: fmt-check vet test lint ## Run every gate CI runs

.PHONY: run
run: build ## Build and run against configs/llmgw.yaml
	./bin/$(BINARY) -config configs/llmgw.yaml

.PHONY: clean
clean:
	rm -rf bin dist

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
