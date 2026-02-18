# Usage Guide

## Main Flow

1. Open training hub: `go run ./cmd/mircogpt-tui`
2. Configure variables in `Train` tab
3. Start training (`s`)
4. Review metrics in `Monitor`
5. Inspect artifacts in `Runs` and `Models`
6. Test checkpoints directly in `Chat` tab

## Train Tab Controls

- `j/k` or arrows: move selection
- `e` or `enter`: edit selected value
- `space`: cycle bool/choice value
- `1/2/3`: apply presets
- `s`: start training
- `x`: stop training

## Chat Tab Controls

- `enter`: send prompt
- `p`: toggle checkpoint path edit
- `L`: use latest checkpoint path
- `[` `]`: adjust temperature
- `-` `=`: adjust max new tokens

## Key Commands

- Train directly: `go run .`
- Validate dataset: `go run . validate-dataset`
- One-shot chat: `go run . chat-once models/latest_checkpoint.json "prompt"`
