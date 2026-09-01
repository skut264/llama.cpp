#!/usr/bin/env bash
# Gate: measured tok/s, drafting ON vs OFF, on one prompt. >=512 tokens, median
# of TRIALS runs. Also reports accept_rate / multiple / drafting_enabled (ON).
#
# ON and OFF trials are INTERLEAVED (on,off,on,off,...) rather than run in two
# blocks, so both conditions share the same thermal state — a fair ratio on a
# passively-cooled laptop where a 3-4 min block would otherwise let one condition
# run hotter (and thus slower) than the other. Still median-of-5 per condition.
#
#   BIN=... MODEL=... ./bench_toks.sh PROMPT_FILE [N] [TRIALS] [NGL] [C] [T]
#
# Env COOLDOWN=<seconds> (default 0) idles between trials. On a passively/lightly
# cooled laptop, sustained back-to-back decode heat-soaks the SoC and throttles the
# clock, which inflates run-to-run variance and can drag the ON/OFF median off its
# true value. A few seconds of COOLDOWN keeps every trial starting from a similar
# thermal state; leave it 0 for the plain reproduce command.
set -euo pipefail
: "${BIN:?}"; : "${MODEL:?}"
P="$1"; N="${2:-512}"; TRIALS="${3:-5}"; NGL="${4:-99}"; C="${5:-2048}"; T="${6:-4}"
COOLDOWN="${COOLDOWN:-0}"

median() { printf '%s\n' "$@" | sort -n | awk '{a[NR]=$0} END{print a[int((NR+1)/2)]}'; }

one_run() {  # $1 = EDGEML_DRAFT_OFF value; echoes tok/s of a single trial
  EDGEML_QUIET=1 EDGEML_DRAFT_OFF="$1" "$BIN" -m "$MODEL" -f "$P" -n "$N" -c "$C" -ngl "$NGL" -t "$T" 2>&1 \
    | awk '/EDGEML decode_tok_s/{print $NF}'
  [ "$COOLDOWN" -gt 0 ] && sleep "$COOLDOWN" || true
}

# one ON run for the accept/multiple/guard stats (also reports the gate value)
statsrun="$(EDGEML_QUIET=1 "$BIN" -m "$MODEL" -f "$P" -n "$N" -c "$C" -ngl "$NGL" -t "$T" 2>&1)"
acc="$(echo "$statsrun"  | awk '/EDGEML accept_rate/{print $NF}')"
mult="$(echo "$statsrun" | awk '/EDGEML multiple/{print $NF}')"
den="$(echo "$statsrun"  | awk '/EDGEML drafting_enabled/{print $NF}')"
dis="$(echo "$statsrun"  | awk '/EDGEML disabled_at/{print $NF}')"
gate="$(echo "$statsrun" | awk '/EDGEML min_score/{print $NF}')"
dstp="$(echo "$statsrun" | awk '/EDGEML draft_steps/{print $NF}')"

on_vals=(); off_vals=()
for i in $(seq 1 "$TRIALS"); do
  on_vals+=("$(one_run 0)")     # interleaved: ON then OFF each trial
  off_vals+=("$(one_run 1)")
done
on_med="$(median "${on_vals[@]}")"; off_med="$(median "${off_vals[@]}")"
speedup="$(awk -v a="$on_med" -v b="$off_med" 'BEGIN{printf (b>0)?"%.3f":"nan", a/b}')"

echo   "prompt        : $(basename "$P")   (n=$N, trials=$TRIALS, ngl=$NGL, min_score=${gate})"
echo   "OFF tok/s     : ${off_vals[*]}   -> median ${off_med}"
echo   "ON  tok/s     : ${on_vals[*]}   -> median ${on_med}"
echo   "speedup       : ${speedup}x  (ON/OFF median, interleaved trials)"
echo   "accept_rate   : ${acc}"
echo   "multiple      : ${mult}  (committed / forward_calls)"
echo   "draft_steps   : ${dstp}   (steps that issued a wide draft batch)"
echo   "drafting_en   : ${den}   disabled_at=${dis}"
