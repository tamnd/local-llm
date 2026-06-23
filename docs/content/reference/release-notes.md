---
title: "Release notes"
description: "How llmgw releases are cut and where to find the per-version changelog."
weight: 30
---

A release is cut by pushing a version tag. GoReleaser takes it from there: it
builds the binaries and publishes the platform archives, the Linux packages
(`.deb`, `.rpm`, `.apk`), a container image, and the Homebrew and Scoop entries.
The version on the tag is stamped into the binary, so `llmgw -version` reports the
release it came from.

The per-version changelog, with the commits behind each release, lives on the
[GitHub releases page](https://github.com/tamnd/local-llm/releases).
