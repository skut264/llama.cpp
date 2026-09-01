// EdgeML prompt-lookup proposer — native C++ port of the reference Python
// package `edgeml/draft` (ngram.py + propose.py + the EMA no-regression guard
// from loop.py). Header-only, depends only on llama.h for `llama_token` and the
// C++ standard library. No ggml/core changes; lives entirely under examples/.
//
// Algorithm (ported verbatim from the reference):
//   * NgramIndex: an incremental polynomial rolling-hash index over the running
//     token stream. For each n in `ngrams` (longest first, e.g. 3 then 2) it maps
//     a length-n token window -> the buffer positions where that window ENDS.
//     Appending a token updates one hash per order (O(#ngrams)); a lookup is a
//     single dict hit (O(1)) — never a scan of the context. Hash collisions are
//     harmless: every match is re-verified against the actual tokens, so a
//     collision costs at most a wasted compare, never a wrong proposal.
//   * LookupProposer::propose(): hashes only the short context tail, does one
//     lookup per order, picks the single best candidate by match_length x recency
//     (a bounded sliding-max), and copies that candidate's continuation (<= D).
//   * SpecGuard: EMA accept-rate + the HARD no-regression rule. If the EMA of
//     accept_rate stays below 1/D for `no_regress_tokens` consecutive committed
//     tokens (after a warmup), drafting is DISABLED for the rest of the session
//     (worst case then == plain greedy + one flag check).
//
// Correctness note: the proposer only affects SPEED. Whatever it drafts is
// verified against the target model's greedy argmax, so a wrong draft only wastes
// a forward pass — the emitted tokens are bit-identical to plain greedy decode.

#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <string>

namespace edgeml {

// Rolling-hash parameters. MOD is a Mersenne prime (~2^61); BASE is an odd
// golden-ratio constant. Hash quality affects only speed, not correctness.
static const uint64_t NG_MOD  = ((uint64_t)1 << 61) - 1;
static const uint64_t NG_BASE = 0x9E3779B1ull;

static inline uint64_t ng_mulmod(uint64_t a, uint64_t b) {
    return (uint64_t)(((unsigned __int128)a * (unsigned __int128)b) % NG_MOD);
}

// Incremental rolling-hash index of the observed token stream (positions where
// each length-n window ends), longest order first.
class NgramIndex {
public:
    NgramIndex(std::vector<int> ngrams = {3, 2}, int max_positions_per_key = 64) {
        // dedup + sort descending (longest first)
        std::sort(ngrams.begin(), ngrams.end(), std::greater<int>());
        ngrams.erase(std::unique(ngrams.begin(), ngrams.end()), ngrams.end());
        ngrams_ = ngrams;
        nmax_   = ngrams_.empty() ? 0 : ngrams_.front();
        cap_    = max_positions_per_key;
        const size_t k = ngrams_.size();
        h_.assign(k, 0);
        bpow1_.assign(k, 1);
        index_.assign(k, {});
        for (size_t i = 0; i < k; ++i) {
            // bpow1_[i] = BASE^(n-1) mod MOD
            uint64_t p = 1;
            for (int e = 0; e < ngrams_[i] - 1; ++e) p = ng_mulmod(p, NG_BASE);
            bpow1_[i] = p;
        }
    }

    int  nmax() const { return nmax_; }
    size_t size() const { return buf_.size(); }
    const std::vector<int> & orders() const { return ngrams_; }
    const std::vector<llama_token> & buffer() const { return buf_; }

    // Append one token and index every newly completed window (O(#orders)).
    void observe_one(llama_token t) {
        const int p = (int) buf_.size();
        buf_.push_back(t);
        const uint64_t tv = (uint64_t)(uint32_t) t % NG_MOD;
        for (size_t i = 0; i < ngrams_.size(); ++i) {
            const int n = ngrams_[i];
            uint64_t h = h_[i];
            if (p >= n) {  // window full: drop oldest, shift up, add newest
                const uint64_t oldest = (uint64_t)(uint32_t) buf_[p - n] % NG_MOD;
                const uint64_t term   = ng_mulmod(oldest, bpow1_[i]);
                const uint64_t h1     = (h + NG_MOD - term) % NG_MOD;  // (h - term) mod MOD
                h = (ng_mulmod(h1, NG_BASE) + tv) % NG_MOD;
            } else {       // window still filling
                h = (ng_mulmod(h, NG_BASE) + tv) % NG_MOD;
            }
            h_[i] = h;
            if (p >= n - 1) {  // a full n-window ends at p
                std::vector<int32_t> & lst = index_[i][h];
                lst.push_back(p);
                if ((int) lst.size() > cap_) lst.erase(lst.begin());  // keep most-recent cap
            }
        }
    }

    void observe(const std::vector<llama_token> & toks) {
        for (llama_token t : toks) observe_one(t);
    }

    // Plain polynomial hash of a window (must equal the incremental hash).
    static uint64_t hash_seq(const llama_token * w, int n) {
        uint64_t h = 0;
        for (int i = 0; i < n; ++i) h = (ng_mulmod(h, NG_BASE) + ((uint64_t)(uint32_t) w[i] % NG_MOD)) % NG_MOD;
        return h;
    }

    // End-positions p (most-recent first) whose length-n window equals
    // tail[len-n:] and that have a continuation (p < last). Collision-safe.
    void lookup(const std::vector<llama_token> & tail, int n, int max_cand, std::vector<int32_t> & out) const {
        out.clear();
        const int tlen = (int) tail.size();
        if (n > tlen || n < 1) return;
        const int oi = order_index(n);
        if (oi < 0) return;
        const llama_token * w = tail.data() + (tlen - n);
        const uint64_t key = hash_seq(w, n);
        auto it = index_[oi].find(key);
        if (it == index_[oi].end()) return;
        const std::vector<int32_t> & lst = it->second;
        const int last = (int) buf_.size() - 1;
        for (int i = (int) lst.size() - 1; i >= 0; --i) {  // recency-ordered
            const int p = lst[i];
            if (p >= last) continue;                        // no continuation (incl. current end)
            const int base = p - n + 1;
            bool eq = true;
            for (int k = 0; k < n; ++k) { if (buf_[base + k] != w[k]) { eq = false; break; } }
            if (eq) { out.push_back(p); if ((int) out.size() >= max_cand) break; }
        }
    }

    // match length = n plus how many tokens BEFORE the window also agree
    // with the context (bounded by cap so scoring stays O(1)).
    int score_back(int p, int n, const std::vector<llama_token> & tail, int cap) const {
        int ml = n;
        int i = p - n;
        int j = (int) tail.size() - n - 1;
        while (ml < cap && i >= 0 && j >= 0 && buf_[i] == tail[j]) { ++ml; --i; --j; }
        return ml;
    }

    // up to d tokens that historically followed the window ending at p.
    void continuation(int p, int d, std::vector<llama_token> & out) const {
        out.clear();
        const int start = p + 1;
        for (int off = 0; off < d && start + off < (int) buf_.size(); ++off) out.push_back(buf_[start + off]);
    }

private:
    int order_index(int n) const {
        for (size_t i = 0; i < ngrams_.size(); ++i) if (ngrams_[i] == n) return (int) i;
        return -1;
    }

    std::vector<int> ngrams_;
    int nmax_ = 0;
    int cap_  = 64;
    std::vector<llama_token> buf_;
    std::vector<uint64_t> h_;      // rolling hash per order
    std::vector<uint64_t> bpow1_;  // BASE^(n-1) per order
    std::vector<std::unordered_map<uint64_t, std::vector<int32_t>>> index_;
};

// Prompt-lookup draft proposer: observe()/propose() over an NgramIndex.
class LookupProposer {
public:
    LookupProposer(int D = 8, std::vector<int> ngrams = {3, 2}, int back_cap = 32, int max_cand = 8)
        : index_(ngrams), D_max_(D), back_cap_(back_cap), max_cand_(max_cand) {
        need_ = index_.nmax() + back_cap_;   // only this many trailing tokens are ever read
    }

    void observe_one(llama_token t) { index_.observe_one(t); }
    void observe(const std::vector<llama_token> & toks) { index_.observe(toks); }
    int  tail_needed() const { return need_; }
    size_t indexed() const { return index_.size(); }
    // score (match_length x recency, i.e. score_back) of the draft emitted by the
    // most recent propose(); -1 if that call drafted nothing. For telemetry only.
    int  last_score() const { return last_score_; }

    // Return <= k draft token ids (appended to `out`); empty out = no draft.
    // O(1) in the size of the index: hashes only the last `need_` tokens.
    //
    // `min_score` is the CONFIDENCE GATE (the primary no-regression mechanism):
    // a candidate is only drafted if its match score (n + how many earlier tokens
    // also agree, capped) is >= min_score. On a real target a wrong D-wide draft
    // costs ~1.5x a single decode, so drafting on weak/coincidental matches (the
    // bulk of novel-prose "matches") regresses wall-clock even though the output
    // stays bit-exact. Gating them out makes that step fall back to a width-1
    // forward == exact baseline cost, so drafting can never make novel text slower.
    // min_score<=n_min (=2) disables the gate (draft on any match, the naive path).
    void propose(const std::vector<llama_token> & inp, int k, std::vector<llama_token> & out, int min_score = 0) {
        out.clear();
        last_score_ = -1;
        const int d = std::min(k, D_max_);
        if (d <= 0 || index_.size() == 0) return;
        const int tlen = (int) inp.size();
        if (tlen == 0) return;
        const int from = tlen > need_ ? tlen - need_ : 0;
        tail_.assign(inp.begin() + from, inp.end());  // small (<= need_ tokens)
        std::vector<int32_t> cands;
        for (int n : index_.orders()) {               // longest match first (3, then 2)
            if ((int) tail_.size() < n) continue;
            index_.lookup(tail_, n, max_cand_, cands);
            if (cands.empty()) continue;
            int best_p = cands[0], best_s = -1;        // cands are most-recent first
            for (int p : cands) {
                const int s = index_.score_back(p, n, tail_, back_cap_);
                if (s > best_s) { best_s = s; best_p = p; }  // ties keep most-recent
            }
            if (best_s < min_score) continue;          // confidence gate: skip weak matches
            index_.continuation(best_p, d, out);
            if (!out.empty()) { last_score_ = best_s; return; }
        }
        out.clear();
    }

private:
    NgramIndex index_;
    int D_max_;
    int back_cap_;
    int max_cand_;
    int need_;
    int last_score_ = -1;
    std::vector<llama_token> tail_;
};

// EMA accept-rate + HARD no-regression guard (ported from SpecDecodeLoop).
class SpecGuard {
public:
    SpecGuard(int D = 8, int ema_window = 64, int no_regress_tokens = 128, long warmup_calls = 64)
        : D_(D), no_regress_(no_regress_tokens), warmup_(warmup_calls) {
        alpha_     = 2.0 / (ema_window + 1);
        threshold_ = 1.0 / (double) D_;
        ema_       = 1.0;  // optimistic start: no premature disable
    }

    bool drafting_enabled() const { return drafting_; }
    long disabled_at()      const { return disabled_at_; }
    double ema()            const { return ema_; }
    double threshold()      const { return threshold_; }
    long verify_calls()     const { return verify_calls_; }
    long drafted()          const { return drafted_; }
    long accepted()         const { return accepted_; }
    double accept_rate()    const { return drafted_ ? (double) accepted_ / (double) drafted_ : 0.0; }

    // Record one verify: n_draft drafted tokens, n_acc accepted, n_commit committed.
    // (Called only when a real draft was verified, i.e. n_draft >= 1 — matching the
    // Python reference, where pure greedy steps do not touch the EMA.)
    void record(long n_draft, long n_acc, long n_commit, long committed_total) {
        ++verify_calls_;
        drafted_  += n_draft;
        accepted_ += n_acc;
        const double sample = n_draft > 0 ? (double) n_acc / (double) n_draft : 0.0;
        ema_ = alpha_ * sample + (1.0 - alpha_) * ema_;
        if (verify_calls_ >= warmup_ && ema_ < threshold_) {
            below_streak_ += n_commit;
            if (below_streak_ >= no_regress_) {
                if (drafting_ && disabled_at_ < 0) disabled_at_ = committed_total;
                drafting_ = false;
            }
        } else {
            below_streak_ = 0;
        }
    }

private:
    int    D_;
    long   no_regress_;
    long   warmup_;
    double alpha_;
    double threshold_;
    double ema_;
    long   below_streak_ = 0;
    bool   drafting_     = true;
    long   verify_calls_ = 0;
    long   drafted_      = 0;
    long   accepted_     = 0;
    long   disabled_at_  = -1;
};

} // namespace edgeml
