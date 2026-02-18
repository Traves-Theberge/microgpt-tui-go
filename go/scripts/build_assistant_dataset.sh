#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN="$ROOT/assistant_dataset_train.v8.jsonl"
EVAL="$ROOT/assistant_dataset_eval.jsonl"
ACTIVE_TRAIN="$ROOT/assistant_dataset_train.jsonl"
ACTIVE_EVAL="$ROOT/assistant_dataset_eval.jsonl"

: > "$TRAIN"
: > "$EVAL"

json_escape() {
  local s="$1"
  s=${s//\\/\\\\}
  s=${s//"/\\"}
  printf '%s' "$s"
}

emit_knowledge() {
  local id="$1" text="$2"
  printf '{"id":"%s","record_type":"knowledge","text":"%s"}\n' "$id" "$(json_escape "$text")" >> "$TRAIN"
}

emit_memory() {
  local id="$1" text="$2"
  printf '{"id":"%s","record_type":"memory","text":"%s"}\n' "$id" "$(json_escape "$text")" >> "$TRAIN"
}

emit_qa() {
  local id="$1" q="$2" a="$3"
  printf '{"id":"%s","record_type":"qa","question":"%s","answer":"%s"}\n' "$id" "$(json_escape "$q")" "$(json_escape "$a")" >> "$TRAIN"
}

emit_chat() {
  local id="$1" in="$2" out="$3"
  printf '{"id":"%s","record_type":"chat","input":"%s","output":"%s"}\n' "$id" "$(json_escape "$in")" "$(json_escape "$out")" >> "$TRAIN"
}

emit_traj() {
  local id="$1" task="$2" a1="$3" a2="$4" a3="$5" a4="$6" result="$7"
  printf '{"id":"%s","record_type":"trajectory","task":"%s","actions":["%s","%s","%s","%s"],"result":"%s"}\n' \
    "$id" "$(json_escape "$task")" "$(json_escape "$a1")" "$(json_escape "$a2")" "$(json_escape "$a3")" "$(json_escape "$a4")" "$(json_escape "$result")" >> "$TRAIN"
}

emit_pref() {
  local id="$1" prompt="$2" chosen="$3" rejected="$4"
  printf '{"id":"%s","record_type":"preference","prompt":"%s","chosen":"%s","rejected":"%s"}\n' \
    "$id" "$(json_escape "$prompt")" "$(json_escape "$chosen")" "$(json_escape "$rejected")" >> "$TRAIN"
}

# Phrase banks for diversity
intents=(
  "plan my day" "prioritize backlog" "recover after delay" "prepare weekly review"
  "draft status update" "structure end-of-day recap" "handle uncertainty"
  "choose next action" "reduce context switching" "sequence project milestones"
  "triage incoming tasks" "prepare stakeholder message"
)

contexts=(
  "for a product launch" "for engineering work" "for operations" "for client delivery"
  "for personal planning" "for a high-pressure week" "for a blocked sprint"
  "for a cross-team effort" "for documentation work" "for implementation mode"
)

constraints=(
  "with limited time" "with one blocker" "with unclear requirements"
  "with dependency risk" "with shifting priorities" "with low energy"
  "with urgent requests" "with strict deadlines"
)

styles=(
  "concise and direct" "checklist-first" "summary then detail"
  "risk-aware" "action-first" "decision-focused"
)

first_steps=(
  "identify the highest-impact outcome" "select one unblock action" "define top three priorities"
  "set a short focus block" "capture assumptions explicitly" "sequence immediate tasks"
)

risk_phrases=(
  "dependency timing risk" "scope ambiguity" "handoff delay" "overloaded queue"
  "context-switching cost" "insufficient validation"
)

mitigations=(
  "confirm constraints early" "reduce scope to essentials" "set owner and checkpoint"
  "publish revised order" "protect focus window" "run quick validation before commit"
)

# Targets (balanced + conversation-heavy)
K=2000
M=2000
Q=3000
C=6000
T=2000
P=3000

# knowledge
for i in $(seq 1 "$K"); do
  a=${intents[$(( i % ${#intents[@]} ))]}
  b=${contexts[$(( (i*3) % ${#contexts[@]} ))]}
  c=${constraints[$(( (i*5) % ${#constraints[@]} ))]}
  d=${styles[$(( (i*7) % ${#styles[@]} ))]}
  e=${first_steps[$(( (i*11) % ${#first_steps[@]} ))]}
  emit_knowledge "k$(printf '%05d' "$i")" "Operational guideline $i: To $a $b $c, use a $d approach and first $e."
done

# memory
for i in $(seq 1 "$M"); do
  a=${styles[$(( i % ${#styles[@]} ))]}
  b=${first_steps[$(( (i*2) % ${#first_steps[@]} ))]}
  r=${risk_phrases[$(( (i*3) % ${#risk_phrases[@]} ))]}
  emit_memory "m$(printf '%05d' "$i")" "Preference signal $i: Keep responses $a, include next actions, and call out $r with mitigation. Start by $b."
done

# qa
for i in $(seq 1 "$Q"); do
  a=${intents[$(( i % ${#intents[@]} ))]}
  b=${contexts[$(( (i*2) % ${#contexts[@]} ))]}
  c=${constraints[$(( (i*3) % ${#constraints[@]} ))]}
  s=${styles[$(( (i*5) % ${#styles[@]} ))]}
  r=${risk_phrases[$(( (i*7) % ${#risk_phrases[@]} ))]}
  m=${mitigations[$(( (i*11) % ${#mitigations[@]} ))]}
  q="How should I $a $b $c in iteration $i?"
  ans="Use a $s plan: define top priorities, execute one immediate step, track $r, and $m."
  emit_qa "q$(printf '%05d' "$i")" "$q" "$ans"
done

# chat
for i in $(seq 1 "$C"); do
  a=${intents[$(( i % ${#intents[@]} ))]}
  b=${contexts[$(( (i*2) % ${#contexts[@]} ))]}
  c=${constraints[$(( (i*3) % ${#constraints[@]} ))]}
  s=${styles[$(( (i*5) % ${#styles[@]} ))]}
  f=${first_steps[$(( (i*7) % ${#first_steps[@]} ))]}
  r=${risk_phrases[$(( (i*11) % ${#risk_phrases[@]} ))]}
  m=${mitigations[$(( (i*13) % ${#mitigations[@]} ))]}
  in="Help me $a $b $c."
  out="Summary: use a $s plan. Priorities: choose highest-impact task, then unblock one dependency, then lock next checkpoint. First step: $f. Risk: $r. Mitigation: $m."
  emit_chat "c$(printf '%05d' "$i")" "$in" "$out"
done

# trajectory
for i in $(seq 1 "$T"); do
  a=${intents[$(( i % ${#intents[@]} ))]}
  b=${contexts[$(( (i*2) % ${#contexts[@]} ))]}
  c=${constraints[$(( (i*3) % ${#constraints[@]} ))]}
  f=${first_steps[$(( (i*5) % ${#first_steps[@]} ))]}
  r=${risk_phrases[$(( (i*7) % ${#risk_phrases[@]} ))]}
  m=${mitigations[$(( (i*11) % ${#mitigations[@]} ))]}
  task="Execute workflow $i to $a $b $c"
  emit_traj "t$(printf '%05d' "$i")" "$task" "Collect current context" "Rank priorities by impact" "$f" "Validate and publish next step" "Result: plan delivered with risk $r and mitigation $m."
done

# preference
for i in $(seq 1 "$P"); do
  a=${intents[$(( i % ${#intents[@]} ))]}
  s=${styles[$(( (i*2) % ${#styles[@]} ))]}
  r=${risk_phrases[$(( (i*3) % ${#risk_phrases[@]} ))]}
  m=${mitigations[$(( (i*5) % ${#mitigations[@]} ))]}
  prompt="Respond to request $i about how to $a"
  chosen="Use a $s response: summary first, concrete priorities, one immediate next action, risk $r, and mitigation $m."
  rejected="There are many possibilities and it depends, maybe decide later."
  emit_pref "p$(printf '%05d' "$i")" "$prompt" "$chosen" "$rejected"
done

# Build eval v2: 60 per type = 360 total
emit_eval_line() {
  printf '%s\n' "$1" >> "$EVAL"
}
for i in $(seq 1 60); do
  emit_eval_line "{\"id\":\"ek$(printf '%04d' "$i")\",\"record_type\":\"knowledge\",\"text\":\"Eval knowledge $i: summary first then actions.\"}"
  emit_eval_line "{\"id\":\"em$(printf '%04d' "$i")\",\"record_type\":\"memory\",\"text\":\"Eval memory $i: keep replies concise and actionable.\"}"
  emit_eval_line "{\"id\":\"eq$(printf '%04d' "$i")\",\"record_type\":\"qa\",\"question\":\"Eval question $i: how should I prioritize now?\",\"answer\":\"Select high-impact work first, then unblock and checkpoint.\"}"
  emit_eval_line "{\"id\":\"ec$(printf '%04d' "$i")\",\"record_type\":\"chat\",\"input\":\"Eval chat request $i\",\"output\":\"Summary, priorities, next action, risk, mitigation.\"}"
  emit_eval_line "{\"id\":\"et$(printf '%04d' "$i")\",\"record_type\":\"trajectory\",\"task\":\"Eval trajectory task $i\",\"actions\":[\"Collect context\",\"Rank\",\"Act\",\"Review\"],\"result\":\"Eval trajectory result $i\"}"
  emit_eval_line "{\"id\":\"ep$(printf '%04d' "$i")\",\"record_type\":\"preference\",\"prompt\":\"Eval preference prompt $i\",\"chosen\":\"Concise actionable response\",\"rejected\":\"Vague delayed response\"}"
done

cp "$TRAIN" "$ACTIVE_TRAIN"
cp "$EVAL" "$ACTIVE_EVAL"

echo "built: $TRAIN"
echo "built: $EVAL"
wc -l "$TRAIN" "$EVAL"
