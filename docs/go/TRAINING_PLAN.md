# Conversational Assistant Training Plan

## Goal

Train a conversational assistant model with consistent style and practical responses.

## Operating Model

Use the TUI for training so monitoring and logs are always on.

```bash
cd go
go run ./cmd/mircogpt-tui
```

## Presets

- `fast`: smoke validation run
- `balanced`: default everyday run
- `max`: slower but stronger run

## Run Checklist

1. `./checks.sh`
2. Launch TUI
3. Select preset (`1/2/3`)
4. Start training (`s`)
5. Watch step metrics + system usage
6. Review generated samples and logs

## Quality Gates

1. Dataset schema passes (`go run . validate-dataset`)
2. Training loss trends down
3. Generated replies stay concise and actionable
4. No malformed output spikes in last 20% of steps
5. RAM/CPU remain stable through run

## Iteration Loop

1. Tag weak outputs (vague, off-topic, verbose)
2. Add corrective `chat` + `qa` + `preference` records
3. Retrain with `balanced`
4. Compare logs and sample quality

## Definition of Done

- Schema clean
- Stable training run in TUI
- Better conversational quality vs previous dataset version
- Logs and metrics archived for comparison
