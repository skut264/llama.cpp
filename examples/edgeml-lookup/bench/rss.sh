#!/usr/bin/env bash
# Gate: no memory growth vs baseline. Peak RSS (maximum resident set size) with
# drafting ON vs OFF, measured by /usr/bin/time -l (macOS). The proposer adds only
# a rolling-hash index (a few tokens' worth of ints), so ON should be within +2%.
#
#   BIN=... MODEL=... ./rss.sh PROMPT_FILE [N] [NGL] [C] [T]
set -euo pipefail
: "${BIN:?}"; : "${MODEL:?}"
P="$1"; N="${2:-512}"; NGL="${3:-99}"; C="${4:-2048}"; T="${5:-4}"

peak_rss() {  # $1 = EDGEML_DRAFT_OFF; echoes peak RSS in bytes
  local off="$1" log; log="$(mktemp)"
  EDGEML_QUIET=1 EDGEML_DRAFT_OFF="$off" /usr/bin/time -l "$BIN" -m "$MODEL" -f "$P" -n "$N" -c "$C" -ngl "$NGL" -t "$T" >/dev/null 2>"$log"
  awk '/maximum resident set size/{print $1}' "$log"
  rm -f "$log"
}

off_b="$(peak_rss 1)"; on_b="$(peak_rss 0)"
delta="$(awk -v a="$on_b" -v b="$off_b" 'BEGIN{printf "%+.3f", (b>0)?100.0*(a-b)/b:0}')"
echo "OFF peak RSS  : ${off_b} bytes ($(awk -v x="$off_b" 'BEGIN{printf "%.1f", x/1048576}') MiB)"
echo "ON  peak RSS  : ${on_b} bytes ($(awk -v x="$on_b" 'BEGIN{printf "%.1f", x/1048576}') MiB)"
echo "delta         : ${delta}%  (ON vs OFF; gate: <= +2%)"
