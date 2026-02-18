# microgpt-go

A tiny GPT-style training demo in Go with one unified training hub TUI.

## Quick Start

```bash
cd /home/traves/Development/6.GroupProjects/microgpt/go
go mod tidy
./checks.sh
```

## Unified MircoGPT-tui (Charmbracelet)

```bash
go run ./cmd/mircogpt-tui
```

Tabs:
- `Train`: full editable config dashboard (all train/runtime vars)
- `Monitor`: live metrics + logs
- `Runs`: recent run logs
- `Models`: saved checkpoints
- `Chat`: integrated chat tester (no separate TUI needed)

Core keys:
- `tab` / `shift+tab` (or `l` / `h`): switch tabs
- `j` / `k`: move field selection (Train tab)
- `e` or `enter`: edit selected field (Train tab)
- `space`: cycle selected bool/choice field (Train tab)
- `1` `2` `3`: apply preset (`fast`, `balanced`, `max`)
- `s`: start training
- `x`: stop training
- `r`: refresh runs/models lists
- `c`: clear logs/chat (current tab)
- `q`: quit

Chat tab keys:
- `enter`: send prompt
- `p`: toggle checkpoint path edit mode
- `L`: use latest checkpoint path
- `[` `]`: decrease/increase chat temperature
- `-` `=`: decrease/increase max new tokens

## Checkpoints

Training auto-saves a checkpoint at the end of each run:
- `go/models/checkpoint_<timestamp>.json`
- `go/models/latest_checkpoint.json`

You can also force an output path:

```bash
MODEL_OUT_PATH=models/my_checkpoint.json go run .
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

One-shot chat from checkpoint:

```bash
go run . chat-once models/latest_checkpoint.json "Help me prioritize my day"
```

## Dataset Format

Use one JSON object per line with required `id` and `record_type`.

Supported `record_type` values:
- `knowledge`
- `memory`
- `qa`
- `chat`
- `trajectory`
- `preference`

Use tracked example datasets:

```bash
cp datasets/examples/assistant_dataset_example_personal_agent.jsonl assistant_dataset_train.jsonl
go run . validate-dataset
```

## Docs

- `../docs/go/INSTALLATION.md`
- `../docs/go/USAGE.md`
- `../docs/go/EXAMPLE_DATASETS.md`
- `../docs/go/TRAINING_PLAN.md`
- `../docs/go/TRAINING_HUB_GUIDE.md`
- `../docs/go/DATASET_GUIDE.md`
