#!/usr/bin/env bash
# Gate: with greedy decode, the spec binary's output token ids are IDENTICAL to
# the greedy baseline (drafting OFF) — and the baseline is cross-checked against
# stock llama-cli text separately (see README). Diffs token ids, not text.
#
#   BIN=.../llama-edgeml-lookup MODEL=.../model.gguf ./bitexact.sh [N] [NGL]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${BIN:?set BIN to llama-edgeml-lookup}"; : "${MODEL:?set MODEL to a .gguf}"
N="${1:-128}"; NGL="${2:-99}"; C="${3:-2048}"; T="${4:-4}"
tmp="$(mktemp -d)"; pass=0; total=0
printf '%-10s %-8s %s\n' "prompt" "n_ids" "result"
for p in "$HERE"/prompts/p??.txt "$HERE"/prompts/prose_novel.txt; do  # p01..p10 + prose_novel = 11 distinct (p??=2 chars, so it does NOT re-match prose_novel)
  name="$(basename "$p")"; total=$((total+1))
  EDGEML_QUIET=1 EDGEML_DUMP_IDS="$tmp/on.ids"  "$BIN" -m "$MODEL" -f "$p" -n "$N" -c "$C" -ngl "$NGL" -t "$T" >/dev/null 2>&1
  EDGEML_QUIET=1 EDGEML_DRAFT_OFF=1 EDGEML_DUMP_IDS="$tmp/off.ids" "$BIN" -m "$MODEL" -f "$p" -n "$N" -c "$C" -ngl "$NGL" -t "$T" >/dev/null 2>&1
  nid="$(wc -l < "$tmp/on.ids" | tr -d ' ')"
  if diff -q "$tmp/on.ids" "$tmp/off.ids" >/dev/null; then
    printf '%-10s %-8s %s\n' "$name" "$nid" "IDENTICAL"; pass=$((pass+1))
  else
    printf '%-10s %-8s %s\n' "$name" "$nid" "DIFFER  <-- FAIL"
    diff "$tmp/on.ids" "$tmp/off.ids" | head
  fi
done
rm -rf "$tmp"
echo "----"
echo "bit-exact: $pass/$total prompts IDENTICAL (spec == greedy)"
[ "$pass" -eq "$total" ]
