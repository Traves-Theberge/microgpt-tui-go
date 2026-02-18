# Example Datasets

Tracked GitHub-ready examples live in:

- `go/datasets/examples/assistant_dataset_example_minimal.jsonl`
- `go/datasets/examples/assistant_dataset_example_personal_agent.jsonl`
- `go/datasets/examples/assistant_dataset_example_eval.jsonl`

## Use An Example

```bash
cd /home/traves/Development/6.GroupProjects/microgpt/go
cp datasets/examples/assistant_dataset_example_personal_agent.jsonl assistant_dataset_train.jsonl
go run . validate-dataset
```

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
