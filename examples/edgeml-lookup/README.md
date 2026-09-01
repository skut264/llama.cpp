# edgeml-lookup — native prompt-lookup speculative decoding for llama.cpp

A self-contained llama.cpp example (`llama-edgeml-lookup`) that speeds up greedy
decoding on **repetitive / code-like** text with **zero quality loss** and **zero
extra model memory** (no draft model), and provably **does not regress** on novel
text. It is a native C++ port of the EdgeML reference proposer (`edgeml/draft`).

## What it is

Speculative decoding without a second model. Each step:

1. **Propose** — a rolling-hash n-gram index over the prompt + generated suffix
   (orders n=3 then n=2) finds where the current context tail recently occurred and
   copies the literal continuation as a draft of up to `D` (default 8) tokens.
   Candidates are fused by **match_length × recency** (a bounded sliding-max). This
   is O(1) in context length — only the last ~35 tokens are ever inspected.
2. **Verify** — one batched forward over the `D+1` draft slots; longest-prefix
   **greedy acceptance** (llama.cpp's proven accept loop). A wrong draft costs a
   forward pass, never a token — so the output is **bit-identical to plain greedy**.
3. **Guard** — an EMA of the accept-rate; if it stays below `1/D` for 128
   consecutive committed tokens (after a warmup), drafting **disables for the rest
   of the session**. Worst case then equals the baseline plus one flag check.

The realized speedup comes from bandwidth amortization: on memory-bound decode,
verifying `D+1` tokens in one forward streams the weights once instead of `D+1`
times, so `multiple = committed / forward_calls` tokens are produced per weight
pass.

## How it relates to what llama.cpp already ships

llama.cpp at tag `b10615` already has substantial self-speculation infrastructure,
and this example deliberately **reuses** rather than reinvents it:

* Bit-exact greedy acceptance and the KV bookkeeping come straight from the
  upstream `examples/lookup/lookup.cpp` verify loop — we did not rewrite them.
* Upstream already has count/frequency-based n-gram lookup (`common/ngram-cache`)
  and five self-speculative `common_speculative_type`s (`ngram-simple`,
  `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`, `ngram-cache`).

What this example adds that is **not** in-tree:

* A **positional** prompt-lookup proposer scored by **match_length × recency**
  (literal-continuation copy), distinct from the existing count/key-value schemes.
* The **hard no-regression guard** (EMA `< 1/D` for 128 committed tokens → disable).
  None of the upstream implementations self-disable on adversarial input; this is
  the guarantee that makes drafting *free* when it cannot help.
* A metrics block matching the EdgeML reference: `accept_rate` (EMA),
  `multiple = committed/forward_calls`, `drafting_enabled`, `disabled_at`.

## Honest results (Qwen2.5-1.7B-Instruct Q4_K_M, M1, Metal)

See `VALIDATION.txt` for the full tables with the exact command behind every
number. Headline (measured; no interpolation):

| workload | OFF tok/s | ON tok/s | ON/OFF | accept_rate | multiple | drafting |
|----------|----------:|---------:|-------:|------------:|---------:|:--------:|
| repetitive / code | 42.73 | 64.71 | **1.514×** | 0.664 | 2.51 | on |
| novel prose (control) | 43.00 | 43.56 | **1.013×** | 0.333 | 1.02 | on |

Qwen2.5-1.7B-Instruct Q4_K_M, M1, Metal, `-n 512`, greedy, interleaved median-of-5.
Repetitive is **1.5×** faster; novel prose is **at baseline** (the confidence gate
suppresses weak matches — only 3 of ~500 steps draft anything). Peak RSS delta
**+0.47%** (median of 3, `/usr/bin/time -l`; worst-case single-sample pairing +3.98%,
OFF-side noise). With the gate **disabled** (`EDGEML_MIN_SCORE=0`) novel regresses
(the reason the gate exists) — see VALIDATION.txt. 7B-class run: **UNMEASURED**
(only the 1.7B model is available locally).

* **Bit-exact:** spec token ids == greedy token ids on 11/11 prompts.
* **Self-test:** proposer exact; propose latency well under 50 µs on an 8k index.
* Repetitive/code and novel-prose tok/s (ON vs OFF), accept_rate, multiple, and
  peak-RSS delta are recorded in VALIDATION.txt.

## Known limits (honest)

* **Greedy only.** This example decodes with argmax. The bit-exactness guarantee is
  a greedy-vs-greedy statement. (The reference algorithm also supports sampled
  decode via the standard speculative sampler, but that is out of scope here.)
* **Single sequence.** One sequence at a time (`seq_id 0`); no batched parallel
  decoding.
* **Speedup is workload-dependent.** The gain tracks how much the output repeats
  its own recent context (code, boilerplate, structured text). On novel prose the
  guard keeps it at ~baseline — that is the design, not a bug.
* **`multiple` is an upper bound on wall-clock speedup**, not equal to it: a batched
  forward of `D+1` tokens is cheaper *per token* than `D+1` single decodes but not
  free, so tok/s speedup < `multiple`.
* **7B-class model UNMEASURED here.** Only `qwen-1.7b.gguf` is available locally
  (HuggingFace is network-blocked in this environment); the larger run is marked
  UNMEASURED in VALIDATION.txt rather than estimated.
* **No ggml/core changes.** Everything is in `examples/edgeml-lookup/` plus a single
  `add_subdirectory` line — by design, so it is a clean, low-risk patch.

## Files

```
examples/edgeml-lookup/
├── edgeml-lookup.cpp     # the binary: greedy verify loop + proposer + guard + stats
├── lookup-proposer.h     # header-only: NgramIndex, LookupProposer, SpecGuard
├── CMakeLists.txt
├── BUILD.md              # exact build + run + reproduce commands
├── README.md             # this file
├── VALIDATION.txt        # measured tables; every number carries its command
└── bench/
    ├── bitexact.sh       # spec ids == greedy ids over the 10+ prompts
    ├── bench_toks.sh     # tok/s ON vs OFF, >=512 tok, median of 5
    ├── rss.sh            # peak RSS ON vs OFF (/usr/bin/time -l)
    └── prompts/          # code_repetitive, prose_novel, p01..p10
```
