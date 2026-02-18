# microgpt-go

`microgpt-go` is a compact Go training project for building a personal assistant model from JSONL data, with a unified terminal UI (`MircoGPT-tui`) for configuration, training, monitoring, artifacts, and chat testing.

## What You Get

- Character-level GPT-style training loop in Go
- Strict dataset schema validation (`id` + `record_type`)
- Unified TUI dashboard: config, monitor, runs, models, chat
- Auto checkpointing after training
- Built-in system monitoring (CPU, RAM, process RSS)
- Live logs and trend graphs

## Quick Start

```bash
cd /home/traves/Development/6.GroupProjects/microgpt/go
go mod tidy
./checks.sh
go run ./cmd/mircogpt-tui
```

## Core Workflow

1. Open TUI: `go run ./cmd/mircogpt-tui`
2. In `Train` tab, set variables or apply a preset (`1/2/3`)
3. Start training with `s`
4. Watch metrics in `Monitor`
5. Review outputs in `Runs` and `Models`
6. Test response behavior in `Chat`

## TUI Overview

Tabs:
- `Train`: full editable variable list with contextual guidance
- `Monitor`: live training metrics, system stats, logs, graphs
- `Runs`: recent run logs
- `Models`: saved checkpoints
- `Chat`: integrated inference testing against checkpoints

Global keys:
- `tab` / `shift+tab` (or `l` / `h`): switch tabs
- `s`: start training
- `x`: stop training
- `r`: refresh run/model lists
- `c`: clear current tab log/chat view
- `q`: quit

Train tab keys:
- `j/k` or arrows: move field selection
- `e` or `enter`: edit selected value
- `space`: cycle bool/choice values
- `1`: apply `fast` preset
- `2`: apply `balanced` preset
- `3`: apply `max` preset

Chat tab keys:
- `enter`: send prompt
- `p`: toggle checkpoint path edit mode
- `L`: auto-load `models/latest_checkpoint.json`
- `[` `]`: decrease/increase chat temperature
- `-` `=`: decrease/increase max new tokens

## Configuration Variables

All are editable in `Train` tab.

Dataset and model shape:
- `DATASET_PATH`
- `N_LAYER`
- `N_EMBD`
- `N_HEAD`
- `BLOCK_SIZE`
- `NUM_STEPS`

Optimizer:
- `LEARNING_RATE`
- `BETA1`
- `BETA2`
- `EPS_ADAM`

Generation:
- `TEMPERATURE`
- `SAMPLE_COUNT`

Runtime and logging:
- `TRAIN_DEVICE` (`cpu` or `gpu` request)
- `METRIC_INTERVAL`
- `LOG_LEVEL` (`info` or `debug`)
- `VERBOSE`

Outputs:
- `MODEL_OUT_PATH`

Notes:
- `N_EMBD` must be divisible by `N_HEAD`.
- Current compute path is CPU. If `gpu` is requested, training logs fallback to CPU.

## Presets

Available presets in TUI:
- `fast`: quick smoke run
- `balanced`: recommended default
- `max`: heavier run

Presets set a base config. You can still edit any field after applying one.

## Dataset Format

Training expects JSONL (one object per line).

Required on each row:
- `id` (unique)
- `record_type`

Supported `record_type` values:
- `knowledge`
- `memory`
- `qa`
- `chat`
- `trajectory`
- `preference`

Example dataset files (GitHub-safe):
- `datasets/examples/assistant_dataset_example_minimal.jsonl`
- `datasets/examples/assistant_dataset_example_personal_agent.jsonl`
- `datasets/examples/assistant_dataset_example_eval.jsonl`

Use an example:

```bash
cp datasets/examples/assistant_dataset_example_personal_agent.jsonl assistant_dataset_train.jsonl
go run . validate-dataset
```

## CLI Commands

Train directly:

```bash
go run .
```

Validate dataset:

```bash
go run . validate-dataset
```

One-shot inference from checkpoint:

```bash
go run . chat-once models/latest_checkpoint.json "Help me prioritize my day"
```

Run full local checks:

```bash
./checks.sh
```

## Artifacts and Paths

Logs:
- `logs/tui_train_<...>.log`
- `logs/tui_system_metrics_<...>.csv`
- `logs/train_latest.log`

Models:
- `models/checkpoint_<timestamp>.json`
- `models/latest_checkpoint.json`

Tracked examples:
- `datasets/examples/*.jsonl`

Ignored (via gitignore):
- runtime logs
- generated checkpoints
- local large dataset variants

## Architecture (High Level)

- `main.go`:
  - dataset parsing/validation
  - training loop
  - checkpoint save/load
  - `chat-once` inference mode
- `cmd/mircogpt-tui/main.go`:
  - unified dashboard
  - process orchestration (`go run .`)
  - metrics parsing and system sampling
  - integrated chat panel

## Known Limitations

- Character-level modeling (not subword tokenization)
- CPU-first implementation (GPU request accepted, then fallback)
- Educational codebase, not production-scale throughput

## Troubleshooting

TUI seems stuck:
- Ensure you are on latest code and rerun `./checks.sh`
- Open `Monitor` tab and verify process status/logs
- Stop with `x`, then start again with `s`

Validation fails:
- Run `go run . validate-dataset`
- Fix the first reported line-level schema error

No chat output:
- Confirm checkpoint exists in `models/latest_checkpoint.json`
- In `Chat` tab, press `L` to load latest path
- Lower temperature and retry

## Documentation Index

- `../docs/go/README.md`
- `../docs/go/INSTALLATION.md`
- `../docs/go/USAGE.md`
- `../docs/go/TRAINING_HUB_GUIDE.md`
- `../docs/go/DATASET_GUIDE.md`
- `../docs/go/EXAMPLE_DATASETS.md`
- `../docs/go/TRAINING_PLAN.md`
