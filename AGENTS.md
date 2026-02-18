# Repository Guidelines

## Project Structure & Module Organization
- `go/` is the runnable project root for the Go implementation.
- Core training/runtime logic lives in `go/main.go`; annotated reference logic is in `go/main_commented.go`.
- Terminal UI entrypoint is `go/cmd/mircogpt-tui/main.go`.
- User documentation is in `docs/go/` (install, usage, datasets, monitoring).
- Runtime outputs are written to `go/logs/` and `go/models/` (ignored by Git except `go/models/.gitkeep`).
- Example datasets and helpers live under `go/datasets/examples/` and `go/scripts/`.

## Build, Test, and Development Commands
- `cd go && go mod tidy`: sync module dependencies.
- `cd go && ./checks.sh`: full local validation (format, build, dataset validation, JSON sanity, smoke run).
- `cd go && go run ./cmd/mircogpt-tui`: launch the interactive TUI.
- `cd go && go run .`: run direct training flow without TUI.
- `cd go && go run . validate-dataset [path]`: validate dataset schema and required fields.
- `cd go && go run . gpu-check`: print GPU/CUDA readiness info.

## Coding Style & Naming Conventions
- Use standard Go formatting; run `gofmt -w` on changed Go files (also enforced by `./checks.sh`).
- Keep packages and files lowercase; use `camelCase` for local vars and `PascalCase` for exported symbols.
- Prefer explicit, descriptive names for training/config fields (match existing env-style naming in UI and docs).
- Keep CLI/TUI commands and log labels consistent with existing terms (`Train`, `Monitor`, `Logs`, `Runs`, `Models`, `Chat`).

## Testing Guidelines
- Primary gate is `go/checks.sh`; run it before opening a PR.
- There are currently no committed `*_test.go` files; add focused Go unit tests for new pure logic when practical.
- For runtime changes, include a reproducible smoke check (example: `timeout 6s go run .`).
- For dataset-related changes, always run `go run . validate-dataset`.

## Commit & Pull Request Guidelines
- Follow the existing commit style: short, imperative subject with a scope/prefix (examples: `feat: ...`, `docs: ...`, `chore: ...`, `tui: ...`).
- Keep commits logically grouped (code, docs, refactor separated when possible).
- PRs should include: purpose, key changes, validation steps run (commands), and terminal screenshots for TUI-visible changes.
- Link related issues/tasks and note any dataset, model, or log-path impacts.
