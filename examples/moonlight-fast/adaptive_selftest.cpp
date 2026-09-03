// Model-free self-test for the EdgeML "Moonlight-fast" adaptive draft-width
// controller (edgeml::AdaptiveWidth in ../edgeml-lookup/lookup-proposer.h).
//
// The full edgeml-lookup example needs cmake + the llama.cpp libraries to link.
// The adaptive-width POLICY, however, is header-only integer logic with no llama
// or ggml dependency, so it can be verified in isolation with plain clang++:
//
//     c++ -std=c++17 -O2 -I. -I../edgeml-lookup adaptive_selftest.cpp -o adaptive_selftest
//     ./adaptive_selftest
//
// (`-I.` picks up the llama.h shim in this directory for the header's single
//  `llama_token` dependency; see llama.h here.)
//
// It asserts the pre-registered policy exactly as wired into the decode loop:
//   * starts at the floor (D=2), never below;
//   * escalates +step only after commit_step committed tokens AND ema>=hi, and
//     never past the ceiling;
//   * de-escalates -step IMMEDIATELY on a single verify with accept-rate < lo,
//     never below the floor, and resets the escalation clock;
//   * a strong EMA below the escalation clock does nothing (clock gating);
//   * a ceiling == floor pins the width (the CPU / -ot bit-exact case);
//   * per-width attribution counts the verify at the width it was drafted at.
//
// Exit code 0 == all checks pass; nonzero == a policy regression.

#include "lookup-proposer.h"   // resolved via -I../edgeml-lookup

#include <cstdio>
#include <vector>
#include <map>

using edgeml::AdaptiveWidth;

static int g_fail = 0;
static void check(const char * name, bool ok) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) ++g_fail;
}

// Drive `n` "good" verifies at the current width: all drafts accepted, EMA high.
// Returns the max width observed. `committed` is advanced by (width+1) per verify.
static int feed_good(AdaptiveWidth & a, int n, long & committed, double ema = 1.0) {
    int mx = a.width();
    for (int i = 0; i < n; ++i) {
        const int w = a.width();
        const long commit = w + 1;              // w accepts + 1 bonus token
        committed += commit;
        a.record(/*n_draft=*/w, /*n_acc=*/w, /*n_commit=*/commit, ema, committed);
        if (a.width() > mx) mx = a.width();
    }
    return mx;
}

int main() {
    printf("adaptive-width self-test (no model)\n");

    // (1) Starts narrow at the floor.
    {
        AdaptiveWidth a(/*floor=*/2, /*ceil=*/8);
        check("(1) starts at floor D=2", a.width() == 2);
        check("(1) floor/ceil reported", a.floor_width() == 2 && a.ceil_width() == 8);
    }

    // (2) Sustained high EMA escalates 2->4->6->8 and STOPS at the ceiling.
    {
        AdaptiveWidth a(2, 8, /*commit_step=*/64, /*hi=*/0.80, /*lo=*/0.50, /*step=*/2);
        long committed = 0;
        const int mx = feed_good(a, 400, committed);   // plenty of commits past 3x64
        check("(2) escalated up to ceiling 8", a.width() == 8);
        check("(2) never exceeded ceiling", mx == 8);
        check("(2) exactly 3 escalations (2->4->6->8)", a.escalations() == 3);
        check("(2) zero de-escalations on all-good", a.deescalations() == 0);
    }

    // (3) One weak verify (accept-rate < lo) de-escalates IMMEDIATELY by one step.
    {
        AdaptiveWidth a(2, 8, 64, 0.80, 0.50, 2);
        long committed = 0;
        feed_good(a, 400, committed);                  // -> width 8
        check("(3) pre-condition width==8", a.width() == 8);
        // a single bad batch: 8 drafted, 0 accepted (inst accept 0.0 < 0.50)
        committed += 1;
        a.record(/*n_draft=*/8, /*n_acc=*/0, /*n_commit=*/1, /*ema=*/1.0, committed);
        check("(3) one bad verify -> width 6", a.width() == 6);
        check("(3) counted one de-escalation", a.deescalations() == 1);
        // three more bad batches walk it down to the floor and stop there
        for (int i = 0; i < 3; ++i) {
            committed += 1;
            a.record(a.width(), 0, 1, 1.0, committed);
        }
        check("(3) de-escalates down to floor 2", a.width() == 2);
        // a further bad batch at the floor must NOT go below and must NOT count
        const long dn_before = a.deescalations();
        committed += 1;
        a.record(a.width(), 0, 1, 1.0, committed);
        check("(3) floor is a hard stop (width stays 2)", a.width() == 2);
        check("(3) no de-escalation counted at floor", a.deescalations() == dn_before);
    }

    // (4) accept-rate exactly == lo does NOT de-escalate ('< lo' is strict).
    {
        AdaptiveWidth a(2, 8, 64, 0.80, 0.50, 2);
        long committed = 0;
        feed_good(a, 100, committed);                  // escalate to >=4
        const int w = a.width();
        check("(4) pre-condition width>=4", w >= 4);
        // 4 drafted, 2 accepted -> inst = 0.50, which is NOT < 0.50
        committed += 3;
        a.record(/*n_draft=*/4, /*n_acc=*/2, /*n_commit=*/3, /*ema=*/1.0, committed);
        check("(4) accept==lo holds width", a.width() == w);
    }

    // (5) Escalation-clock gating: a strong EMA before commit_step does nothing,
    //     and a WEAK ema never escalates no matter how many tokens commit.
    {
        AdaptiveWidth a(2, 8, /*commit_step=*/64, 0.80, 0.50, 2);
        long committed = 0;
        // 10 good verifies (~30 commits) < 64 -> still floor
        feed_good(a, 10, committed);
        check("(5) no escalation before commit_step", a.width() == 2 && committed < 64);

        AdaptiveWidth b(2, 8, 64, 0.80, 0.50, 2);
        long c2 = 0;
        for (int i = 0; i < 400; ++i) {                // lots of commits but EMA below hi
            const int w = b.width();
            c2 += w + 1;
            b.record(w, w, w + 1, /*ema=*/0.60, c2);   // 0.60 < hi=0.80
        }
        check("(5) weak EMA never escalates", b.width() == 2 && b.escalations() == 0);
    }

    // (6) ceiling == floor pins the width (the CPU-clamp / -ot bit-exact case:
    //     effective_draft_width returns 2, so ceil=2 and adaptive is a no-op at 2).
    {
        AdaptiveWidth a(/*floor=*/2, /*ceil=*/2);
        long committed = 0;
        const int mx = feed_good(a, 400, committed);
        check("(6) pinned at 2 when ceil==floor", a.width() == 2 && mx == 2);
        check("(6) no escalations when pinned", a.escalations() == 0);
        // ceiling below floor is clamped up to floor, not down
        AdaptiveWidth b(/*floor=*/2, /*ceil=*/1);
        check("(6) ceil<floor clamps ceil up to floor", b.ceil_width() == 2 && b.width() == 2);
    }

    // (7) per-width attribution: each verify is counted at the width it ran at,
    //     and totals reconcile with the number of verifies fed.
    {
        AdaptiveWidth a(2, 8, 64, 0.80, 0.50, 2);
        long committed = 0;
        const int N = 400;
        feed_good(a, N, committed);
        long total_verifies = 0, total_commits = 0;
        for (const auto & kv : a.per_width()) {
            total_verifies += kv.second.verifies;
            total_commits  += kv.second.commits;
        }
        check("(7) per-width verifies sum to N", total_verifies == N);
        check("(7) width 2 bucket is non-empty (ran before first escalation)",
              a.per_width().count(2) && a.per_width().at(2).verifies > 0);
        check("(7) width 8 bucket is non-empty (ran at the ceiling)",
              a.per_width().count(8) && a.per_width().at(8).verifies > 0);
        check("(7) commits accounted (>0)", total_commits > 0);
    }

    // (8) trajectory records the initial state plus every change, in order.
    {
        AdaptiveWidth a(2, 8, 64, 0.80, 0.50, 2);
        long committed = 0;
        feed_good(a, 400, committed);                  // 3 escalations
        committed += 1; a.record(8, 0, 1, 1.0, committed);  // 1 de-escalation
        const auto & tr = a.trajectory();
        bool ok = !tr.empty() && tr.front().reason == 'I' && tr.front().to == 2;
        int e = 0, d = 0;
        for (const auto & t : tr) { if (t.reason == 'E') ++e; if (t.reason == 'D') ++d; }
        check("(8) trajectory starts with initial state", ok);
        check("(8) trajectory has 3 E and 1 D entries", e == 3 && d == 1);
    }

    printf("adaptive-width self-test: %s (%d failure%s)\n",
           g_fail == 0 ? "ALL PASS" : "FAILURES", g_fail, g_fail == 1 ? "" : "s");
    return g_fail == 0 ? 0 : 1;
}
