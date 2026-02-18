# Changelog

All notable changes to `microgpt-go` are documented here.

## [0.0.4] - 2026-02-18

### Changed
- Monitor tab layout/scrolling fixes for small screens:
  - kept `Metric Explorer` pinned on the right with safer width budgeting.
  - added reliable vertical scrolling behavior for long metric lists.
  - reduced clipping/cutoff issues in narrow or short terminals.
- Logs tab rendering fixes:
  - wrapped long log lines to viewport width before rendering.
  - prevented log overflow from pushing header/tabs out of view.
- Train tab responsiveness improvements:
  - better large-screen layout usage.
  - better small-screen stacked layout and scrolling behavior.
  - improved `Field Detail` readability with split guidance formatting.
- Monitor metric cleanup:
  - removed low-value runtime internals from default display (`Go Heap`, `Go Runtime Sys`, `Goroutines`, `GC Count`).

## [0.0.3] - 2026-02-18

### Added
- Monitor Metric Explorer with category switching (`Core`, `Eval`, `System`) and plain-language metric explanations (`what`, `why`, `how to read`).
- Expanded monitoring metrics:
  - learning rate
  - validation perplexity
  - Go heap memory
  - Go runtime system memory
  - goroutines
  - garbage collection count
- Structured run artifacts:
  - `logs/train/`
  - `logs/system/`
  - `logs/eval/`
  - `logs/runs/` run manifest files
- GPU readiness command: `go run . gpu-check`.
- Run metadata manifests (`logs/runs/run_<tag>.txt`) including dataset + config snapshot.

### Changed
- Monitor graphs now use larger historical windows and improved visual rendering.
- Added harmonica-driven smoothing to additional graph series.
- Runs tab now shows grouped artifacts by type (train/system/eval/manifest).
- Default checkpoint naming now uses run metadata:
  - `models/ckpt_<run-tag>_step<steps>_valloss<loss>.json`
- Removed fixed `MODEL_OUT_PATH` default so auto naming is used by default.
- Removed top `MircoGPT-tui` title to reduce header clutter and improve layout.
- Replaced old help clutter with a cleaner context-aware footer command bar.
- Documentation updated end-to-end for monitor UX, run folders, checkpoint naming, and GPU status.

## [0.0.2] - 2026-02-17

### Added
- BPE-first tokenization pipeline with offline loader support.
- Training/validation split with periodic evaluation.
- Early stopping controls (`EARLY_STOP_PATIENCE`, `EARLY_STOP_MIN_DELTA`).
- Best-checkpoint output (`models/best_checkpoint.json`).
- Improved decoding controls (`TOP_K`, `TOP_P`, repetition penalty, min new tokens).
- Expanded TUI config coverage for tokenizer, validation, and decoding parameters.
- Chat viewport improvements: better wrapping, full-height usage, and history scrolling keys.
- Dataset file picker/search in TUI (`Train` tab on `DATASET_PATH`, key `f`).
- Single-source training dataset replacement from Hugging Face (`databricks-dolly-15k`) converted to project JSONL schema.

### Changed
- Default training profile raised for stronger results:
  - `N_LAYER=4`, `N_EMBD=128`, `N_HEAD=8`, `BLOCK_SIZE=256`, `NUM_STEPS=6000`.
  - `TOKENIZER=bpe`, `BPE_ENCODING=cl100k_base`, `TOKEN_VOCAB_SIZE=8192`.
- Runtime generation and `chat-once` now use higher-quality decoding defaults.
- Documentation synchronized with current runtime behavior and controls.
- Default `TRAIN_DEVICE` request changed to `gpu` (with current CPU fallback warning).

### Removed
- Legacy root dataset variants replaced by single active training dataset:
  - `assistant_dataset_eval.jsonl`
  - `assistant_dataset_real_chat.jsonl`
  - `assistant_dataset_seed.jsonl`
  - `assistant_dataset_train.v8.jsonl`
- Stale local binaries cleaned from workspace (`chat-tui`, `train-tui`, etc.).

## [0.0.1] - 2026-02-17 (retroactive)

### Initial baseline (MircoGPT Go)
- Character-level Go GPT training loop with JSONL schema validation.
- Unified terminal dashboard (`mircogpt-tui`) with Train/Monitor/Runs/Models/Chat tabs.
- Live logs, process/system metrics, and checkpoint saving.
- Basic chat testing flow using `chat-once` and `models/latest_checkpoint.json`.
- Preset-based training and local checks via `./checks.sh`.
