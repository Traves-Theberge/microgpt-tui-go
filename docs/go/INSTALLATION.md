# Installation

## Requirements

- Go `1.25+`
- Linux/macOS terminal
- `jq` (used by `./checks.sh`)

## Setup

```bash
cd go
go mod tidy
./checks.sh
```

## First Run

```bash
go run ./cmd/mircogpt-tui
```

In the TUI `Train` tab, use:
- `2` for `coherent-fast` (recommended first real run on CPU)
- `3` for `coherent-max` (long run, best quality target)

## Included Dataset

- Active dataset: `datasets/raw/databricks-dolly-15k.jsonl`
- Source snapshot: `datasets/raw/databricks-dolly-15k.jsonl`

## Optional GPU Check

Run:

```bash
go run . gpu-check
```

This verifies NVIDIA visibility (`nvidia-smi`) and CUDA toolkit presence (`nvcc`), then reports current trainer status (CPU kernels).

## Version

- Current project version: `0.0.4`
- Changelog: `../../go/CHANGELOG.md`
