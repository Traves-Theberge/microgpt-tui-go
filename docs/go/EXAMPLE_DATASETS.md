# Example Datasets

Tracked GitHub-ready examples live in:

- `go/datasets/examples/assistant_dataset_example_minimal.jsonl`
- `go/datasets/examples/assistant_dataset_example_personal_agent.jsonl`
- `go/datasets/examples/assistant_dataset_example_eval.jsonl`

## Use An Example

```bash
cd go
cp datasets/examples/assistant_dataset_example_personal_agent.jsonl datasets/generated/example_personal_agent.jsonl
go run . validate-dataset datasets/generated/example_personal_agent.jsonl
```

For normal `0.0.3` training, keep the default Dolly dataset in place and use examples only for local experiments.

## Schema Reminder

Each JSONL line requires:

- `id` (unique)
- `record_type`

Supported `record_type`:

- `knowledge`
- `memory`
- `qa`
- `chat`
- `trajectory`
- `preference`
