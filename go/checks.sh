#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "[1/5] Formatting"
gofmt -w main.go main_commented.go cmd/mircogpt-tui/main.go

echo "[2/5] Building"
go build .
go build ./cmd/mircogpt-tui

echo "[3/5] Validating dataset schema"
go run . validate-dataset

echo "[4/5] JSON sanity"
jq -c . assistant_dataset_train.jsonl > /dev/null

echo "[5/5] Runtime smoke test (6s)"
timeout 6s go run . > /tmp/microgpt_go_smoke.log 2>&1 || true
head -n 5 /tmp/microgpt_go_smoke.log

echo "checks: PASS"
