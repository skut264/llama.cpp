# Benchmarks — edgeml-lookup

These are the digested result tables. The RAW runs — with the exact command behind
every single number here — live in [VALIDATION.txt](VALIDATION.txt) (Gates 1-6);
nothing below is interpolated or estimated. Base: llama.cpp @ b10615 (commit
f280b2698); model qwen2.5-1.7B-Instruct Q4_K_M; host Apple M1 (Metal + Accelerate).
All tok/s are greedy, `-n 512`, interleaved median-of-5 unless noted.

## Metal (-ngl 99) — primary validated backend (D=8)

| workload | OFF tok/s | ON tok/s | ON/OFF | accept_rate | multiple | drafting |
|---|---|---|---|---|---|---|
| repetitive / code | 42.73 | 64.71 | 1.514x | 0.664 | 2.51 | on |
| novel prose (control) | 43.00 | 43.56 | 1.013x | 0.333 | 1.02 | on |

Peak RSS delta +0.47% (median-of-3, `/usr/bin/time -l`; Gate 5).

## CPU (-ngl 0) — bit-exact by construction (clamped to D=2)

| workload | OFF | ON | ON/OFF | vs target | bit-exact? |
|---|---|---|---|---|---|
| repetitive / code | 46.20 | 54.54 | 1.181x | >=1.2x (misses ~1.6%) | yes |
| novel prose (control) | 47.94 | 46.71 | 0.974x | >=0.98x (misses ~2.6%) | yes |
| repetitive, opt-out D=8 | 47.24 | 62.13 | 1.315x | — | no |

The clamp (`EDGEML_CPU_SAFE_D=2`) is what makes CPU bit-exact; `EDGEML_ALLOW_WIDE_CPU=1`
restores D=8 (1.315x) at the cost of bit-exactness. CPU is compute-bound here, which is
why the bit-exact setting lands just under the targets (see VALIDATION Gate 3b "WHY CPU
misses").

## Bit-exactness — per-backend x per-D matrix

Values are "N/11 IDENTICAL" (from VALIDATION Gate 2b), forcing raw widths via
`EDGEML_ALLOW_WIDE_CPU=1`.

| D | CPU (-ngl 0) | Metal (-ngl 99) |
|---|---|---|
| D=1 | 11/11 | (n/a) |
| D=2 | 11/11 (default clamp) | 11/11 |
| D=3 | 5/11 | 11/11 |
| D=8 | 5/11 | 11/11 |

CPU is bit-exact up to width 3 (D<=2), diverges at width>=4 (D>=3); Metal is exact at
all D. Divergence is deterministic FP reduction-order (a valid greedy stream, no quality
loss), NOT a logic bug. Shipped default clamps CPU to D=2, so shipped CPU is 11/11.

## Cross-hardware trend

**Provenance:** only the Apple M1 row was measured in this environment. The VPS rows were
REPORTED by an external downstream verifier (Linux aarch64, Neoverse-N1, gcc 12, CPU-only)
and recorded in PR skut264/llama.cpp#1 — they were NOT measured here.

| host / model | backend | repetitive | novel | source |
|---|---|---|---|---|
| Apple M1 / qwen2.5-1.7B | Metal (GPU), D=8 | 1.514x | 1.013x | measured here (VALIDATION.txt) |
| VPS aarch64 / qwen2.5-1.5B | CPU, unclamped D=8 | 1.27x | 1.048x | downstream verifier, PR #1 — NOT measured here |
| VPS aarch64 / qwen2.5-1.5B | CPU, clamped D=2 | 1.12x | 1.012x | downstream verifier, PR #1 — NOT measured here |
| VPS aarch64 / qwen 4B | CPU (clamp state as reported) | 1.209x | 1.023x | downstream verifier, PR #1 — NOT measured here |

Across these four rows the speedup is positive on repetitive workloads and roughly
break-even to slightly positive on novel prose, but it is workload- and backend-dependent:
the compute-bound CPU path gains less than the bandwidth-bound Metal path. These rows do
not establish a clean monotonic model-size trend beyond what the four measurements show.
7B-class: UNMEASURED (only the 1.7B model is available locally; HuggingFace is
network-blocked).
