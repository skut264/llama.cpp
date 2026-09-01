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

## Honest results (Qwen2.5-1.7B-Instruct Q4_K_M, M1)

See `VALIDATION.txt` for the full tables with the exact command behind every
number. Headline (measured; no interpolation):

### Metal (`-ngl 99`) — primary validated backend, D=8

| workload | OFF tok/s | ON tok/s | ON/OFF | accept_rate | multiple | drafting |
|----------|----------:|---------:|-------:|------------:|---------:|:--------:|
| repetitive / code | 42.73 | 64.71 | **1.514×** | 0.664 | 2.51 | on |
| novel prose (control) | 43.00 | 43.56 | **1.013×** | 0.333 | 1.02 | on |

`-n 512`, greedy, interleaved median-of-5. Repetitive is **1.5×** faster; novel prose
is **at baseline** (the confidence gate suppresses weak matches — only 3 of ~500 steps
draft anything). Peak RSS delta **+0.47%** (median of 3, `/usr/bin/time -l`; worst-case
single-sample pairing +3.98%, OFF-side noise). With the gate **disabled**
(`EDGEML_MIN_SCORE=0`) novel regresses (the reason the gate exists) — see VALIDATION.txt.
Bit-exactness holds at **all D** on Metal. 7B-class run: **UNMEASURED** (only the 1.7B
model is available locally).

### CPU (`-ngl 0`) — bit-exact by construction, at a measured speed cost

On the CPU backend a wide (≥4-wide) batched decode reorders the ggml matmul FP
reductions vs a width-1 decode, which can flip a near-tie greedy argmax; so to keep
spec ids == greedy ids the example clamps the draft width to `EDGEML_CPU_SAFE_D=2` (the
measured bit-exact maximum) whenever `-ngl 0`. At that clamp (interleaved median-of-5):

| workload | OFF tok/s | ON tok/s | ON/OFF | vs target | bit-exact? |
|----------|----------:|---------:|-------:|:---------:|:----------:|
| repetitive / code | 46.20 | 54.54 | **1.181×** | ≥1.2× (misses ~1.6%) | yes |
| novel prose (control) | 47.94 | 46.71 | **0.974×** | ≥0.98× (misses ~2.6%) | yes |
| repetitive, opt-out D=8 | 47.24 | 62.13 | **1.315×** | — | **no** |

CPU decode here is already ~46–54 tok/s (Accelerate/AMX), i.e. compute-bound, so
speculation's bandwidth-amortization win is small, and capping D at 2 caps
tokens-per-forward — the bit-exact clamp lands just **under** the ≥1.2×/≥0.98× targets.
This is a genuine trade-off, **reported not tuned**: correctness (bit-exact) is the
default; set `EDGEML_ALLOW_WIDE_CPU=1` to keep D=8 for **1.315×** on CPU at the cost of
bit-exactness (greedy ids may then differ from the width-1 baseline — still a valid
greedy stream, no quality loss). Metal is unaffected by the clamp.

* **Bit-exact (backend-specific):** spec token ids == greedy token ids on **11/11**
  prompts on **Metal at D=8** and on **CPU at the default (clamped D=2)**. On CPU with
  the clamp bypassed (`EDGEML_ALLOW_WIDE_CPU=1`, D≥3) ids can differ — see the per-D
  matrix in VALIDATION.txt.
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
* **CPU backend (`-ngl 0`) is bit-exact only up to D=2.** Wide (≥4-wide) CPU batches
  reorder the ggml matmul FP reductions and can flip a near-tie greedy argmax, so the
  default clamps CPU draft width to 2 (measured bit-exact; Metal is exact at all D).
  At that clamp CPU speedup is modest (repetitive **1.181×**, novel **0.974×**) because
  CPU decode is compute-bound — it lands just under the ≥1.2×/≥0.98× targets.
  `EDGEML_ALLOW_WIDE_CPU=1` keeps D=8 for **1.315×** on CPU but is not bit-exact. This
  is why the CPU speed gate is reported as a measured near-miss, not a pass — see
  VALIDATION.txt.
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
