// Minimal llama.h SHIM for the model-free adaptive-width self-test.
//
// lookup-proposer.h intentionally depends on llama.h for exactly one thing: the
// `llama_token` typedef (a 32-bit token id). This shim provides only that, so the
// whole header (NgramIndex + LookupProposer + SpecGuard + AdaptiveWidth) compiles
// and its logic can be exercised with plain clang++ and NO model, NO ggml, and NO
// prebuilt llama libraries. When built inside the real llama.cpp tree the true
// include/llama.h is used instead; this file is only for the standalone test and
// is NOT installed or linked into the actual example binary.
#pragma once
#include <cstdint>
typedef int32_t llama_token;
