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
```

Decoding is **pure greedy** (argmax over the target logits); the emitted tokens
are bit-identical whether drafting is ON or OFF (drafting only changes speed).
Generation produces exactly `-n` tokens (eos is recorded, not a stop condition),
so ON/OFF runs are directly comparable.

## Environment knobs

All extra configuration is via env vars, to keep the patch confined to `examples/`:

| Env var             | Default | Meaning                                                        |
|---------------------|---------|----------------------------------------------------------------|
| `EDGEML_DRAFT_OFF`  | unset   | `1` = greedy baseline (drafting disabled)                      |
| `EDGEML_D`          | `8`     | max draft length D                                             |
| `EDGEML_MIN_SCORE`  | `5`     | confidence gate: only draft matches with score >= this (primary no-regression knob; `<=2` disables it) |
| `EDGEML_WARMUP`     | `64`    | verify calls before the no-regression guard can arm           |
| `EDGEML_NO_REGRESS` | `128`   | consecutive committed tokens under 1/D that disable drafting   |
| `EDGEML_EMA_WINDOW` | `64`    | EMA window for accept_rate                                     |
| `EDGEML_DUMP_IDS`   | unset   | path to write generated token ids (one per line)              |
| `EDGEML_STATS_JSON` | unset   | path to write a one-line JSON stats record                    |
| `EDGEML_QUIET`      | unset   | `1` = do not stream generated text to stdout                  |
| `EDGEML_SELFTEST`   | unset   | `1` = run the model-free proposer self-test and exit          |

## Reproduce the validation tables

```sh
export BIN=$PWD/build/bin/llama-edgeml-lookup
export MODEL=/path/to/qwen-1.7b.gguf
cd examples/edgeml-lookup/bench
./bitexact.sh 128 99          # gate: spec ids == greedy ids over 10 prompts
./bench_toks.sh prompts/code_repetitive.txt 512 5 99   # tok/s ON vs OFF (repetitive)
./bench_toks.sh prompts/prose_novel.txt     512 5 99   # tok/s ON vs OFF (novel control)
./rss.sh        prompts/code_repetitive.txt 512   99   # peak RSS ON vs OFF
```
