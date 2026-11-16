// tt/spec/spec_capi.cpp - a C ABI over Strata's speculation sources (third_party/strata/src/spec/, read-only), for
// tt/spec/strata_spec.py (ctypes). Nothing is reimplemented: every call forwards to the Strata class, so the drafts
// and the policy's picks are Strata's own. Built by tt/spec/build.sh together with those sources.
#include "strata/spec/controller.hpp"
#include "strata/spec/draft_policy.hpp"
#include "strata/spec/draft_source.hpp"
#include "strata/spec/suffix_drafter.hpp"

#include <cstddef>
#include <cstdint>

using strata::spec::Choice;
using strata::spec::Controller;
using strata::spec::CostModel;
using strata::spec::DraftPolicy;
using strata::spec::PromptLookupSource;
using strata::spec::Source;
using strata::spec::SuffixDrafter;

extern "C" {

// ---- SuffixDrafter
void* sd_new(int min_match, int max_match, uint64_t capacity) {
    return new SuffixDrafter(min_match, max_match, (size_t) capacity);
}
void sd_free(void* d) { delete (SuffixDrafter*) d; }
void sd_reset(void* d) { ((SuffixDrafter*) d)->reset(); }
void sd_append(void* d, const int32_t* t, uint64_t n) { ((SuffixDrafter*) d)->append(t, (size_t) n); }
int sd_propose(void* d, int max_k, int32_t* out) { return ((SuffixDrafter*) d)->propose(max_k, out); }
int sd_propose_after(void* d, const int32_t* extra, int n_extra, int max_k, int32_t* out) {
    return ((SuffixDrafter*) d)->propose_after(extra, n_extra, max_k, out);
}
int sd_last_match(void* d) { return ((SuffixDrafter*) d)->last_match(); }
uint64_t sd_size(void* d) { return ((SuffixDrafter*) d)->size(); }

// ---- PromptLookupSource + propose_from_sources (--lookup-chain): the request's own text, no extra sources
int sd_propose_sources(void* d, const int32_t* tail, int n_tail, int n_pending, int max_k, int min_match, int32_t* out,
                       int* match, int* source) {
    PromptLookupSource own(*(SuffixDrafter*) d);
    return strata::spec::propose_from_sources(own, tail, n_tail, n_pending, max_k, min_match, out, match, source);
}

// ---- DraftPolicy
void* dp_new(int max_t, double margin) { return new DraftPolicy(max_t, margin); }
void dp_free(void* p) { delete (DraftPolicy*) p; }
void dp_choose(void* p, int t_mtp, int lookup_k, int match, int* lookup, int* t) {
    const DraftPolicy::Pick k = ((DraftPolicy*) p)->choose(t_mtp, lookup_k, match);
    *lookup = k.lookup ? 1 : 0;
    *t = k.t;
}
void dp_observe(void* p, int lookup, int t, int accepted, int match, double round_ms) {
    ((DraftPolicy*) p)->observe(lookup != 0, t, accepted, match, round_ms);
}
int dp_chain(void* p, int t_mtp, double p_mtp, int k_avail, int match) {
    return ((DraftPolicy*) p)->chain(t_mtp, p_mtp, k_avail, match);
}
void dp_observe_chain(void* p, int t_mtp, int k, int accepted, int match, double round_ms) {
    ((DraftPolicy*) p)->observe_chain(t_mtp, k, accepted, match, round_ms);
}
double dp_chain_rate(void* p, int match) { return ((DraftPolicy*) p)->chain_rate(match); }
double dp_lookup_rate(void* p, int match) { return ((DraftPolicy*) p)->lookup_rate(match); }
double dp_cost_ms(void* p, int t) { return ((DraftPolicy*) p)->cost_ms(t); }

// ---- Controller (plan v0.3 P6's cost-model controller; the serving loop uses DraftPolicy, this is for reference)
// cost: NULL for the defaults, else the 33 doubles dense_ms, dense_ratio[9], cpu_all_miss_ms, hit_rate,
// distinct_ratio[9], extra_use_cost, sync_ms, mtp_draft_ms, lookup_draft_ms, then 9 unused
void* ct_new(const double* cost, double min_gain, double ema) {
    CostModel c;
    if (cost != nullptr) {
        int i = 0;
        c.dense_ms = cost[i++];
        for (double& v : c.dense_ratio) v = cost[i++];
        c.cpu_all_miss_ms = cost[i++];
        c.hit_rate = cost[i++];
        for (double& v : c.distinct_ratio) v = cost[i++];
        c.extra_use_cost = cost[i++];
        c.sync_ms = cost[i++];
        c.mtp_draft_ms = cost[i++];
        c.lookup_draft_ms = cost[i++];
    }
    return new Controller(c, min_gain, ema);
}
void ct_free(void* c) { delete (Controller*) c; }
// out: source (0 none, 1 lookup, 2 mtp), k; expected tokens and tokens per ms in out_d[0], out_d[1]
void ct_choose(void* c, int lookup_available, int lookup_match, int mtp_ready, int* out_i, double* out_d) {
    const Choice ch = ((Controller*) c)->choose(lookup_available, lookup_match, mtp_ready != 0);
    out_i[0] = (int) ch.source;
    out_i[1] = ch.k;
    out_d[0] = ch.expected_tokens;
    out_d[1] = ch.tokens_per_ms;
}
void ct_observe(void* c, int source, int k, double expected, double tpm, int accepted, int lookup_match) {
    Choice ch;
    ch.source = (Source) source;
    ch.k = k;
    ch.expected_tokens = expected;
    ch.tokens_per_ms = tpm;
    ((Controller*) c)->observe(ch, accepted, lookup_match);
}
double ct_mtp_accept(void* c, int position) { return ((Controller*) c)->mtp_accept(position); }
double ct_lookup_accept(void* c, int match) { return ((Controller*) c)->lookup_accept(match); }
double ct_step_ms(void* c, int k, int mtp) { return ((Controller*) c)->cost().step_ms(k, mtp != 0); }

}  // extern "C"
