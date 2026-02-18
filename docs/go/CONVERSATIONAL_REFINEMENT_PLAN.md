# Conversational Refinement Plan v1

## Current State (Measured)

Train dataset (`assistant_dataset_train.v8.jsonl`):
- 18,000 records total
- chat: 6,000
- qa: 3,000
- preference: 3,000
- knowledge: 2,000
- memory: 2,000
- trajectory: 2,000

Eval dataset (`assistant_dataset_eval.jsonl`):
- 360 records (60 per type)

## Practical Delta vs Real Conversational Datasets

Compared with real instruction/chat datasets (Dolly, OASST1, UltraChat), the biggest gap is **semantic diversity**.

Measured uniqueness in v8 baseline synthetic profile:
- chat outputs: 6 unique / 6000 total
- qa answers: 6 unique / 3000 total
- preference chosen: 6 unique / 3000 total

Implication:
- high volume but low conversational variety
- model will overfit response templates
- assistant quality will feel repetitive and rigid

## Target for “Actual Conversational” Quality

Minimum target for next version:
1. At least 35-50% of train set from real or realistic multi-turn conversation artifacts.
2. Chat output uniqueness ratio > 0.25 on sampled records.
3. Preference pairs derived from real edits and rewrites, not only synthetic templates.
4. Eval set with hard cases: ambiguity, conflicting priorities, stress scenarios, follow-up turns.

## Data Source Strategy

Priority order:
1. Real chat logs with your own prompts and preferred responses
2. Real planning/status notes converted into user-assistant exchanges
3. Revision history turned into preference pairs (chosen vs rejected)
4. Synthetic data only to fill coverage gaps

## Conversational Record Design

For multi-turn behavior, encode short history in `chat.input`:

- `chat.input`: include 1-3 previous turns
- `chat.output`: next assistant turn only

Example:

```jsonl
{"id":"c_real_0001","record_type":"chat","input":"User: I missed yesterday's deadline.\nAssistant: Let's reset with essentials.\nUser: I only have 2 hours now, what first?","output":"Start with the highest-impact deliverable slice, then send a revised ETA update. Keep scope minimal for this block."}
```

## Execution Plan

1. Ingest real transcripts into structured JSONL
2. Add 2,000+ real chat-derived records
3. Add 500+ real preference pairs from rewrites
4. Rebalance dataset to keep type coverage while increasing real conversational share
5. Validate schema + run quality checks
6. Recompute uniqueness and move only if thresholds pass

## Gates Before Training

Required pass criteria:
- `./checks.sh` pass
- eval schema pass (`go run . validate-dataset assistant_dataset_eval.jsonl`)
- uniqueness threshold pass on chat/qa/preference sampled outputs
- manual spot-check of 100 random records

## Next Deliverable

- `assistant_dataset_train.v8.jsonl` with real-conversation infusion and improved diversity
- `assistant_dataset_eval.v3.jsonl` with hard conversational cases
