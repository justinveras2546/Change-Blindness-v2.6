#!/usr/bin/env bash
# Build a local binary (no SDL install needed: it is embedded).
set -euo pipefail
cd "$(dirname "$0")"
CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o Change-Blindness .
echo "built ./Change-Blindness — run it with: ./Change-Blindness -s 1"
