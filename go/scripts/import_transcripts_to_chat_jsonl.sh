#!/usr/bin/env bash
set -euo pipefail

# Converts simple transcript files into chat records.
# Input format per file:
#   U: user text
#   A: assistant text
# Repeated turns. Produces records using rolling history windows.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$ROOT/data_sources/conversations"
OUT_FILE="$ROOT/assistant_dataset_real_chat.jsonl"
PREFIX="c_real"

: > "$OUT_FILE"

json_escape() {
  local s="$1"
  s=${s//\\/\\\\}
  s=${s//"/\\"}
  printf '%s' "$s"
}

idn=0

for f in "$SRC_DIR"/*.txt; do
  [ -e "$f" ] || continue

  # Read transcript lines and build rolling turn arrays
  mapfile -t lines < "$f"
  users=()
  assists=()

  for ln in "${lines[@]}"; do
    if [[ "$ln" == U:* ]]; then
      users+=("${ln#U: }")
    elif [[ "$ln" == A:* ]]; then
      assists+=("${ln#A: }")
    fi
  done

  pairs=${#assists[@]}
  if (( ${#users[@]} < pairs )); then
    pairs=${#users[@]}
  fi

  for ((i=0; i<pairs; i++)); do
    # Build up to 2 prior turns as context
    ctx=""
    start=$(( i-2 ))
    if (( start < 0 )); then start=0; fi
    for ((j=start; j<i; j++)); do
      ctx+="User: ${users[$j]}\\nAssistant: ${assists[$j]}\\n"
    done
    input="${ctx}User: ${users[$i]}"
    output="${assists[$i]}"

    idn=$((idn+1))
    printf '{"id":"%s_%05d","record_type":"chat","input":"%s","output":"%s"}\n' \
      "$PREFIX" "$idn" "$(json_escape "$input")" "$(json_escape "$output")" >> "$OUT_FILE"
  done

done

echo "wrote: $OUT_FILE"
wc -l "$OUT_FILE"
