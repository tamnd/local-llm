# Consumed by GoReleaser: it copies the already cross-compiled binary out of the
# build context rather than compiling, so the image build is fast and uses the
# same static binary every other artifact ships.
#
# This image is the pure-Go gateway only. It proxies to inference backends
# (Ollama, llama-server, TabbyAPI, vLLM) reachable over the network; it does not
# bundle CUDA or the in-process cgo engine, which is built on the box itself with
# `make build-llama` against a CUDA-linked libllama.
#
# GoReleaser builds one multi-platform image with buildx and stages each
# platform's binary under a $TARGETPLATFORM directory (e.g. linux/amd64/) in the
# build context, so the COPY line selects the right one through the automatic
# TARGETPLATFORM build arg.
FROM alpine:3.21

ARG TARGETPLATFORM

# ca-certificates so the gateway can reach HTTPS backends; tzdata for sane log
# timestamps.
RUN apk add --no-cache ca-certificates tzdata

COPY $TARGETPLATFORM/llmgw /usr/bin/llmgw

# The data plane. The admin plane stays on loopback inside the container.
EXPOSE 8888

# Mount a config and point the gateway at it:
#
#   docker run --rm -p 8888:8888 \
#     -v "$PWD/llmgw.yaml:/etc/llmgw.yaml:ro" \
#     ghcr.io/tamnd/local-llm -config /etc/llmgw.yaml
#
ENTRYPOINT ["/usr/bin/llmgw"]
