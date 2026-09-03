# examples/moonlight-fast — adaptive draft-width speculative decode

This extends `examples/edgeml-lookup` (EdgeML prompt-lookup speculative decode) with
an **adaptive draft-width** controller for the Moonlight-16B-A3B target on an 8 GB
Mac. It is the Lane B deliverable of the "Moonlight fast" sprint.

## Why adaptive D

Prompt-lookup spec-decode drafts `D` tokens per verify and accepts the prefix that
matches the target model's greedy argmax. The best `D` is workload-dependent:

- **repetitive / code** text sustains wide drafts — `D=8` accepts most of the batch,
  so each expensive target forward commits many tokens (high "multiple");
- **novel prose** barely matches — at `D=8` most drafted tokens are rejected, so the
  wide batched forward is wasted work and wall-clock *regresses* vs a narrow draft.

A single fixed `D` is therefore wrong for a mixed stream. `AdaptiveWidth`
(`../edgeml-lookup/lookup-proposer.h`) floats `D` per stream from the accept-rate EMA
the guard already computes.

Draft width changes **speed only** — every drafted token is verified against the
target's greedy argmax, so the emitted token stream is **bit-identical for any `D`
trajectory** (see the bit-exactness matrix in the pack's VALIDATION.txt).

## Policy (pre-registered — not tuned to a benchmark)

| knob | env | default | meaning |
|------|-----|---------|---------|
| floor | `EDGEML_ADAPTIVE_FLOOR` | 2 | always-safe start width (≥1 real drafted token) |
| ceiling | `EDGEML_D` | 8 | `= effective_draft_width(D_req, cpu_numerics, …)`; the bit-exact max |
| escalate hi | `EDGEML_ADAPTIVE_HI` | 0.80 | widen only while guard EMA ≥ hi |
| commit step | `EDGEML_ADAPTIVE_COMMIT` | 64 | at most one +step per this many committed tokens |
| de-escalate lo | `EDGEML_ADAPTIVE_LO` | 0.50 | shrink immediately if one verify's accept-rate < lo |
| step | `EDGEML_ADAPTIVE_STEP` | 2 | width increment/decrement |
| on/off | `EDGEML_ADAPTIVE` | 1 | `0` pins the classic fixed width `D` (for the per-D matrix) |

**Asymmetric by design:** slow to widen (needs a sustained 0.80 EMA over 64 tokens),
instant to retreat (a single sub-0.50 batch). That bounds the wasted-draft cost of a
workload change to a single wide verify.

### Bit-exactness / the `-ot` path

`effective_draft_width` clamps the ceiling to `EDGEML_CPU_SAFE_D` (=2) whenever the
CPU numeric path is live, because a wide (≥4-wide) CPU batched decode reorders the
FP reductions and can flip a near-tie greedy argmax (measured bit-exact ≤ width 3,
i.e. `D≤2`; Metal is bit-exact at all `D`). Under the throughput lever
`-ngl 99 -ot "exps=CPU"` the routed-expert GEMV runs on the CPU even though
`n_gpu_layers=99`, so **set `EDGEML_OT=1` alongside `-ot`** to re-arm that clamp
(there is no libllama API to detect an `-ot` override from the example). Opt out with
`EDGEML_ALLOW_WIDE_CPU=1` if you want raw speed and can tolerate ids differing from
the width-1 baseline. See RUNBOOK.md in the pack root.

## What builds here vs. what needs cmake

- **`adaptive_selftest.cpp` — builds & runs HERE with plain clang++/g++, no model.**
  The `AdaptiveWidth` policy is header-only integer logic (its only llama dependency
  is the `llama_token` typedef, satisfied by the `llama.h` shim in this directory).

      ./build_selftest.sh

  Verifies floor/ceiling, earned escalation, instant de-escalation, clock gating,
  the pinned (`ceil==floor`) case, per-width attribution, and the trajectory log.
  Exit 0 == all pass.

- **The full example binary (`llama-edgeml-lookup`)** links libllama/libggml and is
  built by the normal llama.cpp cmake flow once these sources are dropped into the
  tree (see BUILD.md in `../edgeml-lookup`). The driver wiring lives in
  `../edgeml-lookup/edgeml-lookup.cpp` (adaptive is ON by default; telemetry prints
  the D trajectory and a per-width accept-rate / multiple table).

## Telemetry

On exit the driver prints (and, with `EDGEML_STATS_JSON=path`, writes as JSON):

    EDGEML adaptive         = 1
    EDGEML D_floor          = 2
    EDGEML D_ceiling        = 8
    EDGEML D_final          = 6
    EDGEML escalations      = 4
    EDGEML deescalations    = 1
    EDGEML D_trajectory     = I:0->2@0 E:2->4@66 E:4->6@131 D:6->4@140 E:4->6@205
    EDGEML per-width [D: verifies commits drafts accepts accept_rate multiple]
    EDGEML   D=2: verifies=... commits=... accept_rate=... multiple=...
    EDGEML   D=4: ...
    EDGEML   D=6: ...
