// edgeml-lookup — native prompt-lookup speculative decoding for llama.cpp.
//
// This is a self-contained example binary (llama-edgeml-lookup). It reuses
// llama.cpp's proven greedy verify/accept + KV bookkeeping (so the emitted tokens
// are bit-identical to plain greedy decode) but replaces the draft source with
// the EdgeML LookupProposer (rolling-hash n-gram index, length x recency fusion)
// and adds the HARD no-regression guard. No ggml/core edits.
//
// Sampling is pure greedy (argmax over the target logits) — no sampler config can
// break bit-exactness. Generation always produces exactly -n tokens (eos is not a
// stop condition here; it is recorded like any other token) so ON vs OFF token
// streams and timings are directly comparable.
//
// Extra knobs (env vars, to keep the patch confined to examples/):
//   EDGEML_DRAFT_OFF=1     run the greedy baseline (drafting disabled) — for ON/OFF
//   EDGEML_D=8             max draft length D (default 8)
//   EDGEML_MIN_SCORE=5     confidence gate: only draft matches whose score
//                          (match_length x recency) >= this. Default 5, chosen from
//                          a measured sweep: novel prose stays >=0.98x (0.994x at 5
//                          vs 0.968x at 4) while repetitive keeps >=1.2x (1.52x).
//                          This is the primary no-regression mechanism; <=2 disables it.
//   EDGEML_WARMUP=64       verify calls before the no-regression guard can arm
//   EDGEML_NO_REGRESS=128  consecutive committed tokens under 1/D that disable drafting
//   EDGEML_EMA_WINDOW=64   EMA window for accept_rate
//   EDGEML_DUMP_IDS=path   write generated token ids (one per line) — for bit-exact diff
//   EDGEML_STATS_JSON=path write a one-line JSON stats record
//   EDGEML_QUIET=1         do not stream generated text to stdout
//   EDGEML_SELFTEST=1      run the model-free proposer self-test and exit

#include "arg.h"
#include "ggml.h"
#include "common.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include "lookup-proposer.h"

#include <algorithm>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// --------------------------------------------------------------------------- //
// small env helpers                                                           //
// --------------------------------------------------------------------------- //

static int env_int(const char * k, int def) {
    const char * v = getenv(k);
    if (!v || !*v) return def;
    return atoi(v);
}
static bool env_flag(const char * k) {
    const char * v = getenv(k);
    return v && *v && !(v[0] == '0' && v[1] == '\0');
}
static std::string env_str(const char * k) {
    const char * v = getenv(k);
    return v ? std::string(v) : std::string();
}

static llama_token greedy_argmax(llama_context * ctx, int idx, int n_vocab) {
    const float * lg = llama_get_logits_ith(ctx, idx);
    llama_token best = 0;
    float bv = lg[0];
    for (int v = 1; v < n_vocab; ++v) {  // '>' keeps the first max — matches numpy/llama greedy
        if (lg[v] > bv) { bv = lg[v]; best = v; }
    }
    return best;
}

// --------------------------------------------------------------------------- //
// model-free self-test of the proposer (gate 3: O(1) propose latency + logic)  //
// --------------------------------------------------------------------------- //

static int run_selftest() {
    using namespace edgeml;
    fprintf(stderr, "edgeml-lookup self-test (no model)\n");
    int fails = 0;

    // (1) repetitive stream: propose must reproduce the true continuation.
    {
        const int N = 8000, P = 251;
        std::vector<llama_token> stream;
        stream.reserve(N);
        for (int i = 0; i < N; ++i) stream.push_back((llama_token)(i % P) + 1);

        LookupProposer prop(8);
        prop.observe(stream);

        int checked = 0, ok = 0;
        for (int t : {300, 1000, 4000, 7990}) {
            std::vector<llama_token> pre(stream.begin(), stream.begin() + t);
            std::vector<llama_token> draft;
            prop.propose(pre, 8, draft);
            const int d = (int) draft.size();
            bool match = d > 0;
            for (int j = 0; j < d; ++j) if (t + j >= N || draft[j] != stream[t + j]) { match = false; break; }
            checked++; ok += match ? 1 : 0;
            if (!match) fprintf(stderr, "  [1] FAIL: repetitive propose at t=%d (drafted %d)\n", t, d);
        }
        fprintf(stderr, "  [1] repetitive continuation exact: %d/%d\n", ok, checked);
        if (ok != checked) fails++;

        // (1b) confidence gate: a long repetitive run has a high match score, so a
        // moderate gate still drafts; an unreachably-high gate suppresses drafting
        // entirely (that step then falls back to a width-1 == baseline forward).
        {
            std::vector<llama_token> pre(stream.begin(), stream.begin() + 4000), draft;
            prop.propose(pre, 8, draft, /*min_score=*/4);
            const int  strong_score = prop.last_score();
            const bool drafts_when_strong = !draft.empty() && strong_score >= 4;
            prop.propose(pre, 8, draft, /*min_score=*/100000);
            const bool gated_when_weak = draft.empty() && prop.last_score() < 0;
            fprintf(stderr, "  [1b] gate: strong->draft(score=%d) %d ; high-gate->empty %d\n",
                    strong_score, (int) drafts_when_strong, (int) gated_when_weak);
            if (!(drafts_when_strong && gated_when_weak)) { fprintf(stderr, "  [1b] FAIL: gate logic\n"); fails++; }
        }
    }

    // (2) propose latency on an 8k-token index must be well under 50 us (no scan).
    {
        const int N = 8000, P = 251;
        std::vector<llama_token> stream;
        stream.reserve(N);
        for (int i = 0; i < N; ++i) stream.push_back((llama_token)((i * 7 + 3) % P) + 1);
        LookupProposer prop(8);
        prop.observe(stream);

        std::vector<double> batch_us;
        std::vector<llama_token> draft;
        volatile long sink = 0;
        for (int b = 0; b < 7; ++b) {
            const int calls = 4000;
            const int64_t t0 = ggml_time_us();
            for (int c = 0; c < calls; ++c) {
                prop.propose(stream, 8, draft);
                sink += (long) draft.size();
            }
            const int64_t t1 = ggml_time_us();
            batch_us.push_back(double(t1 - t0) / calls);
        }
        std::sort(batch_us.begin(), batch_us.end());
        const double median_us = batch_us[batch_us.size() / 2];
        fprintf(stderr, "  [2] propose latency on 8k index: median %.3f us/call (sink=%ld)\n", median_us, (long) sink);
        if (!(median_us < 50.0)) { fprintf(stderr, "  [2] FAIL: propose latency >= 50 us\n"); fails++; }
    }

    // (3) guard: a sustained sub-1/D accept rate must disable drafting for good.
    {
        SpecGuard g(/*D=*/8, /*ema_window=*/64, /*no_regress=*/128, /*warmup=*/64);
        long committed = 0;
        // feed 4000 verify calls of 8-token drafts that accept 0 tokens (worst case).
        for (int i = 0; i < 4000 && g.drafting_enabled(); ++i) { committed += 1; g.record(8, 0, 1, committed); }
        fprintf(stderr, "  [3] guard on starvation: disabled=%d at=%ld ema=%.4f (< thr %.4f)\n",
                (int) !g.drafting_enabled(), g.disabled_at(), g.ema(), g.threshold());
        if (g.drafting_enabled()) { fprintf(stderr, "  [3] FAIL: guard did not disable\n"); fails++; }

        // and a healthy stream must NOT disable.
        SpecGuard g2(8, 64, 128, 64);
        long c2 = 0;
        for (int i = 0; i < 4000; ++i) { c2 += 9; g2.record(8, 8, 9, c2); }
        if (!g2.drafting_enabled()) { fprintf(stderr, "  [3] FAIL: guard disabled on a healthy stream\n"); fails++; }
    }

    fprintf(stderr, "%s\n", fails == 0 ? "SELF-TEST OK" : "SELF-TEST FAILED");
    return fails == 0 ? 0 : 1;
}

// --------------------------------------------------------------------------- //
// main                                                                        //
// --------------------------------------------------------------------------- //

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    if (env_flag("EDGEML_SELFTEST")) {
        return run_selftest();
    }

    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_LOOKUP)) {
        return 1;
    }

    const int   D              = std::max(1, env_int("EDGEML_D", 8));
    const bool  draft_off      = env_flag("EDGEML_DRAFT_OFF");
    const int   min_score      = std::max(0, env_int("EDGEML_MIN_SCORE", 5));
    const int   warmup         = std::max(0, env_int("EDGEML_WARMUP", 64));
    const int   no_regress     = std::max(1, env_int("EDGEML_NO_REGRESS", 128));
    const int   ema_window     = std::max(1, env_int("EDGEML_EMA_WINDOW", 64));
    const bool  quiet          = env_flag("EDGEML_QUIET");
    const std::string dump_ids = env_str("EDGEML_DUMP_IDS");
    const std::string json_out = env_str("EDGEML_STATS_JSON");

    // configure the context to allow up to D+1 logits-producing positions per decode
    params.speculative.draft.n_max = D;
    const auto output_limits = common_speculative_get_output_limits(params.n_batch, params.n_parallel, D);
    params.n_outputs_max = output_limits.total;
    params.n_outputs_max_per_seq = output_limits.per_seq;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (!model || !ctx) {
        LOG_ERR("%s: failed to load model\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> inp = common_tokenize(ctx, params.prompt, true, true);
    if (inp.empty()) {
        LOG_ERR("%s: empty prompt\n", __func__);
        return 1;
    }

    const int n_ctx = llama_n_ctx(ctx);
    const int n_new = params.n_predict > 0 ? params.n_predict : 512;
    if ((int) inp.size() + n_new + D + 4 > n_ctx) {
        LOG_ERR("%s: prompt(%d)+n_predict(%d) exceed n_ctx(%d); raise -c\n",
                __func__, (int) inp.size(), n_new, n_ctx);
        return 1;
    }

    // proposer over the whole prompt + generated suffix
    edgeml::LookupProposer prop(D);
    edgeml::SpecGuard guard(D, ema_window, no_regress, warmup);
    prop.observe(inp);
    bool drafting = !draft_off;

    if (!quiet) {
        for (llama_token id : inp) LOG("%s", common_token_to_piece(ctx, id).c_str());
        fflush(stderr);
    }

    // prefill
    const int n_input = (int) inp.size();
    const auto t_enc_start = ggml_time_us();
    llama_decode(ctx, llama_batch_get_one( inp.data(), n_input - 1));
    llama_decode(ctx, llama_batch_get_one(&inp.back(),           1));
    const auto t_enc_end = ggml_time_us();

    int  n_predict    = 0;
    long forward_calls = 0;   // generation-phase target decodes (the cost metric)
    long committed     = 0;
    int  n_past        = n_input;

    std::vector<llama_token> draft;               // holds the drafted continuation to verify
    std::vector<llama_token> gen_ids;             // for optional id dump
    gen_ids.reserve(n_new);
    llama_batch batch_tgt = llama_batch_init(n_ctx, 0, 1);

    long prev_n_draft   = 0;                       // drafts awaiting verification (from last decode)
    long draft_steps    = 0;                       // steps that issued a non-empty draft (wide forward)
    long empty_proposes = 0;                       // steps drafting-on but proposer returned nothing (width-1)

    const auto t_dec_start = ggml_time_us();
    bool done = false;
    while (!done && n_predict < n_new) {
        long round_acc = 0, round_commit = 0;
        int  i_dft = 0;
        while (true) {
            const llama_token id = greedy_argmax(ctx, i_dft, n_vocab);
            ++n_predict; ++committed; ++round_commit;
            inp.push_back(id);
            prop.observe_one(id);
            gen_ids.push_back(id);
            if (!quiet) { LOG("%s", common_token_to_piece(ctx, id).c_str()); fflush(stdout); }

            const bool stop = (n_predict >= n_new);
            if (i_dft < (int) draft.size() && id == draft[i_dft]) {
                // sampled token matches the drafted token -> accepted
                ++round_acc; ++n_past; ++i_dft;
                if (stop) { done = true; break; }
                continue;
            }
            // mismatch (or ran out of draft): id is the correction / bonus token
            draft.clear();
            draft.push_back(id);
            if (stop) { done = true; break; }
            break;
        }

        // update EMA / guard for the draft that was just verified
        if (prev_n_draft > 0) {
            guard.record(prev_n_draft, round_acc, round_commit, committed);
            if (!guard.drafting_enabled()) drafting = false;
        }
        if (done) break;

        // KV: drop any drafted tokens that were not accepted
        llama_memory_seq_rm(llama_get_memory(ctx), 0, n_past, -1);

        // draft the continuation from the proposer (if drafting is enabled)
        if (drafting) {
            std::vector<llama_token> cont;
            prop.propose(inp, D, cont, min_score);
            for (llama_token t : cont) draft.push_back(t);
            if (cont.empty()) ++empty_proposes; else ++draft_steps;
        }
        prev_n_draft = (long) draft.size() - 1;    // exclude the anchor draft[0]

        // one batched forward verifies the whole draft (draft.size() logits slots)
        common_batch_clear(batch_tgt);
        for (size_t i = 0; i < draft.size(); ++i) {
            common_batch_add(batch_tgt, draft[i], n_past + (int) i, { 0 }, true);
        }
        llama_decode(ctx, batch_tgt);
        ++forward_calls;
        ++n_past;
        draft.erase(draft.begin());
    }
    const auto t_dec_end = ggml_time_us();

    const double dec_s   = (t_dec_end - t_dec_start) / 1e6;
    const double tok_s   = dec_s > 0 ? n_predict / dec_s : 0.0;
    const double mult    = forward_calls > 0 ? (double) committed / (double) forward_calls : 0.0;
    const char * mode    = draft_off ? "baseline" : "spec";

    LOG("\n\n");
    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n",
            n_input, (t_enc_end - t_enc_start) / 1e6, n_input / ((t_enc_end - t_enc_start) / 1e6));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, dec_s, tok_s);
    LOG_INF("\n");
    LOG_INF("EDGEML mode             = %s\n", mode);
    LOG_INF("EDGEML D                = %d\n", D);
    LOG_INF("EDGEML min_score        = %d\n", min_score);
    LOG_INF("EDGEML n_predict        = %d\n", n_predict);
    LOG_INF("EDGEML forward_calls    = %ld\n", forward_calls);
    LOG_INF("EDGEML committed        = %ld\n", committed);
    LOG_INF("EDGEML draft_steps      = %ld\n", draft_steps);
    LOG_INF("EDGEML empty_proposes   = %ld\n", empty_proposes);
    LOG_INF("EDGEML verify_calls     = %ld\n", guard.verify_calls());
    LOG_INF("EDGEML drafted          = %ld\n", guard.drafted());
    LOG_INF("EDGEML accepted         = %ld\n", guard.accepted());
    LOG_INF("EDGEML accept_rate      = %.4f\n", guard.accept_rate());
    LOG_INF("EDGEML ema              = %.4f\n", guard.ema());
    LOG_INF("EDGEML multiple         = %.4f\n", mult);
    LOG_INF("EDGEML drafting_enabled = %d\n", (int) guard.drafting_enabled());
    LOG_INF("EDGEML disabled_at      = %ld\n", guard.disabled_at());
    LOG_INF("EDGEML decode_tok_s     = %.4f\n", tok_s);

    if (!dump_ids.empty()) {
        std::ofstream f(dump_ids);
        for (llama_token id : gen_ids) f << id << "\n";
        LOG_INF("EDGEML wrote %zu ids -> %s\n", gen_ids.size(), dump_ids.c_str());
    }
    if (!json_out.empty()) {
        std::ofstream f(json_out);
        f << "{"
          << "\"mode\":\"" << mode << "\","
          << "\"D\":" << D << ","
          << "\"min_score\":" << min_score << ","
          << "\"n_predict\":" << n_predict << ","
          << "\"forward_calls\":" << forward_calls << ","
          << "\"committed\":" << committed << ","
          << "\"draft_steps\":" << draft_steps << ","
          << "\"empty_proposes\":" << empty_proposes << ","
          << "\"verify_calls\":" << guard.verify_calls() << ","
          << "\"drafted\":" << guard.drafted() << ","
          << "\"accepted\":" << guard.accepted() << ","
          << "\"accept_rate\":" << guard.accept_rate() << ","
          << "\"ema\":" << guard.ema() << ","
          << "\"multiple\":" << mult << ","
          << "\"drafting_enabled\":" << (int) guard.drafting_enabled() << ","
          << "\"disabled_at\":" << guard.disabled_at() << ","
          << "\"decode_tok_s\":" << tok_s
          << "}\n";
    }

    llama_batch_free(batch_tgt);
    llama_backend_free();
    LOG("\n");
    return 0;
}
