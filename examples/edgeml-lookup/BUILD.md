# Building `llama-edgeml-lookup`

Native prompt-lookup speculative decoding for llama.cpp. Self-contained example;
**no ggml/core changes**.

## Pinned base

* Upstream: `https://github.com/ggml-org/llama.cpp`
* Tag: **`b10615`** (commit `f280b2698f...`)

The patch series in `patches/` applies cleanly on top of that exact tag.

## Prerequisites

* A C++17 compiler (tested: Apple clang on macOS arm64 / M1)
* CMake ≥ 3.14 and a generator (Unix Makefiles or Ninja)
* A GGUF model (tested: `qwen-1.7b.gguf`, Qwen2.5-1.7B-Instruct Q4_K_M)

## Apply the patch and build

```sh
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout b10615

# apply the EdgeML lookup example (adds examples/edgeml-lookup + 1 line in examples/CMakeLists.txt)
git am /path/to/patches/0001-*.patch      # or: git apply /path/to/patches/*.patch

# configure + build (exact commands used for VALIDATION.txt)
cmake -B build -DGGML_NATIVE=ON
cmake --build build -j --target llama-edgeml-lookup
```

On macOS the Metal and Accelerate backends are detected automatically. The binary
lands at `build/bin/llama-edgeml-lookup`.

> The VALIDATION.txt runs additionally passed `-DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
> -DLLAMA_BUILD_SERVER=OFF` to keep the build minimal; none of these affect the
> example. `-DGGML_NATIVE=ON` is the only flag the task requires.

## Sanity: model-free self-test

Validates the proposer (rolling-hash index, length×recency fusion), the O(1)
propose latency (< 50 µs on an 8k-token index), and the no-regression guard —
with **no model loaded**:

```sh
EDGEML_SELFTEST=1 ./build/bin/llama-edgeml-lookup
# -> SELF-TEST OK
```

## Run

```sh
# speculative (drafting ON)
./build/bin/llama-edgeml-lookup -m qwen-1.7b.gguf -f prompt.txt -n 512 -c 2048 -ngl 99

# greedy baseline (drafting OFF) — identical tokens, one forward per token
EDGEML_DRAFT_OFF=1 ./build/bin/llama-edgeml-lookup -m qwen-1.7b.gguf -f prompt.txt -n 512 -c 2048 -ngl 99

# set the draft length D via the idiomatic speculative flag (EDGEML_D=<n> is an env alias/override)
./build/bin/llama-edgeml-lookup -m qwen-1.7b.gguf -f prompt.txt -n 512 -c 2048 -ngl 99 --spec-draft-n-max 8
```

Decoding is **pure greedy** (argmax over the target logits). On the GPU/Metal
backend the emitted tokens are bit-identical whether drafting is ON or OFF at any
`D` (drafting only changes speed). Generation produces exactly `-n` tokens (eos is
recorded, not a stop condition), so ON/OFF runs are directly comparable.

### CPU backend bit-exactness (`-ngl 0`)

A wide (`>=4`-wide) batched decode on the **ggml CPU matmul path** sums the logit
dot-products in a different order than a width-1 decode. That reordering can flip a
near-tie greedy argmax, so with a large draft width the spec token stream can drift
from the width-1 baseline — still a *valid* greedy stream (no quality loss), but no
longer *bit-identical*. This is floating-point non-associativity, not a logic bug,
and it does **not** occur on Metal (whose reductions match width-1 at all `D`).

To keep "spec ids == greedy ids" true **by construction** on the CPU backend, the
example clamps the draft width to `EDGEML_CPU_SAFE_D` (default **2**, measured
bit-exact) whenever the run offloads nothing to the GPU (`-ngl 0`). Measured on this
repo: CPU is bit-exact up to batch width 3 (`D<=2`) and can diverge at `D>=3`. Set
`EDGEML_ALLOW_WIDE_CPU=1` to keep the full `D` on CPU (faster, but ids may differ
from the baseline). See `VALIDATION.txt` for the per-backend × per-`D` matrix.

## Environment knobs

All extra configuration is via env vars, to keep the patch confined to `examples/`:

| Env var             | Default | Meaning                                                        |
|---------------------|---------|----------------------------------------------------------------|
| `EDGEML_DRAFT_OFF`  | unset   | `1` = greedy baseline (drafting disabled)                      |
| `EDGEML_D`          | `8`     | max draft length D — env alias/override for the `--spec-draft-n-max` CLI flag (clamped on the CPU backend — see below)    |
| `EDGEML_ALLOW_WIDE_CPU` | unset | `1` = on the CPU backend (`-ngl 0`) do **not** clamp D. Faster, but greedy ids may differ from the width-1 baseline (see "CPU backend bit-exactness") |
| `EDGEML_CPU_SAFE_D` | `2`     | max draft width used on the CPU backend (`-ngl 0`) when not overridden; measured bit-exact up to 2. Metal is unaffected (no clamp) |
| `EDGEML_MIN_SCORE`  | `5`     | confidence gate: only draft matches with score >= this (primary no-regression knob; `<=2` disables it) |
| `EDGEML_WARMUP`     | `64`    | verify calls before the no-regression guard can arm           |
| `EDGEML_NO_REGRESS` | `128`   | consecutive committed tokens under 1/D that disable drafting   |
| `EDGEML_EMA_WINDOW` | `64`    | EMA window for accept_rate                                     |
| `EDGEML_DUMP_IDS`   | unset   | path to write generated token ids (one per line)              |
| `EDGEML_STATS_JSON` | unset   | path to write a one-line JSON stats record                    |
| `EDGEML_QUIET`      | unset   | `1` = do not stream generated text to stdout                  |
| `EDGEML_SELFTEST`   | unset   | `1` = run the model-free proposer self-test and exit          |

## Reproduce the validation tables

> `BENCHMARKS.md` has the digested result tables, the per-backend × per-`D`
> bit-exactness matrix, and the cross-hardware trend; `VALIDATION.txt` keeps the
> raw runs with the exact command behind every number.

```sh
export BIN=$PWD/build/bin/llama-edgeml-lookup
export MODEL=/path/to/qwen-1.7b.gguf
cd examples/edgeml-lookup/bench
./bitexact.sh 128 99          # gate: spec ids == greedy ids, Metal   (D=8, no clamp)
./bitexact.sh 128 0           # gate: spec ids == greedy ids, CPU     (D auto-clamped to 2)
./bench_toks.sh prompts/code_repetitive.txt 512 5 99   # tok/s ON vs OFF, Metal (repetitive)
./bench_toks.sh prompts/prose_novel.txt     512 5 99   # tok/s ON vs OFF, Metal (novel control)
COOLDOWN=6 ./bench_toks.sh prompts/code_repetitive.txt 512 5 0   # tok/s ON vs OFF, CPU clamped D=2 (repetitive)
COOLDOWN=6 ./bench_toks.sh prompts/prose_novel.txt     512 5 0   # tok/s ON vs OFF, CPU clamped D=2 (novel control)
./rss.sh        prompts/code_repetitive.txt 512   99   # peak RSS ON vs OFF
```
