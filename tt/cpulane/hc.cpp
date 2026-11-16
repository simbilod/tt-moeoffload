// GLM's mHC parametrization on the host (the card's side in tt/kernels/glm_hc_*.cpp): from the cores'
// partials (the card's D2H page: mixes [nc][32] fp32, then a 16-byte slot a core whose first float is its sum of
// squares) to the 24 parameters (pre [4], post [4], C [dst][src] [16], fp32), in double, llama.cpp's build_hc_pre:
// r = rsqrt(mean of squares + eps); a = r * mixes * scale + base; pre = sigmoid(a[0:4]) + eps; post = 2 sigmoid(a[4:8]);
// comb = the row softmax of a[8:24] as [src][dst], + eps, a column normalization, then (iters - 1) row and column
// normalizations; C = comb transposed. Same arithmetic as tt/glm/g3_hc.py's host_params (numpy), which checks it.
#include <cmath>
#include <cstdint>

extern "C" void cl_hc_params(const float* page, int nc, int d, int iters, const float* scale, const float* base,
                             float* out) {
    constexpr int HC = 4;
    constexpr double EPS_RMS = 1e-5, EPS_HC = 1e-6;
    double mix[24] = {0}, ss = 0;
    for (int c = 0; c < nc; c++) {
        for (int j = 0; j < 24; j++) mix[j] += page[c * 32 + j];
        ss += page[nc * 32 + 4 * c];
    }
    const double r = 1.0 / std::sqrt(ss / (HC * d) + EPS_RMS);
    double a[24];
    for (int j = 0; j < 24; j++) a[j] = r * mix[j] * scale[j < 4 ? 0 : (j < 8 ? 1 : 2)] + base[j];
    auto sig = [](double v) { return 1.0 / (1.0 + std::exp(-v)); };
    for (int j = 0; j < 4; j++) {
        out[j] = static_cast<float>(sig(a[j]) + EPS_HC);
        out[4 + j] = static_cast<float>(2.0 * sig(a[4 + j]));
    }
    double m[HC][HC];
    for (int i = 0; i < HC; i++) {
        double mx = a[8 + 4 * i], s = 0;
        for (int j = 1; j < HC; j++) mx = a[8 + 4 * i + j] > mx ? a[8 + 4 * i + j] : mx;
        for (int j = 0; j < HC; j++) s += (m[i][j] = std::exp(a[8 + 4 * i + j] - mx));
        for (int j = 0; j < HC; j++) m[i][j] = m[i][j] / s + EPS_HC;
    }
    auto col = [&] {
        for (int j = 0; j < HC; j++) {
            const double s = m[0][j] + m[1][j] + m[2][j] + m[3][j] + EPS_HC;
            for (int i = 0; i < HC; i++) m[i][j] /= s;
        }
    };
    col();
    for (int it = 1; it < iters; it++) {
        for (int i = 0; i < HC; i++) {
            const double s = m[i][0] + m[i][1] + m[i][2] + m[i][3] + EPS_HC;
            for (int j = 0; j < HC; j++) m[i][j] /= s;
        }
        col();
    }
    for (int dst = 0; dst < HC; dst++) {
        for (int src = 0; src < HC; src++) out[8 + dst * HC + src] = static_cast<float>(m[src][dst]);
    }
    for (int k = 24; k < 32; k++) out[k] = 0.0f;
}

// GLM's decode indexer on the host: the k pools of highest score among n (k <= n) as ids into out;
// ties to the lower id (a strict, total order), so the set does not depend on the selection's path. The order of out
// is not sorted by score (the attention sums over the pools in any order): first every pool above the threshold's
// bin in id order, then the bin's best in key order.
// Each score maps to an order-preserving unsigned key (negatives inverted, positives' sign bit set). A radix select:
// one pass histograms the top 11 bits and finds the bin holding the k-th key, one pass takes every pool above it and
// collects the bin's few candidates, which are sorted by (key, then lower id). An index array with a comparator
// through score[] took 379 us at 32K pools, nth_element on packed 64-bit keys 170 us.
#include <algorithm>
#include <cstring>
#include <functional>
#include <vector>

extern "C" void cl_top_k(const float* score, int n, int k, uint32_t* out) {
    thread_local std::vector<uint32_t> u;
    thread_local std::vector<uint64_t> cand;
    u.resize(n);
    cand.resize(n);
    uint32_t hist[4][2048] = {};   // four interleaved histograms: a hot bin's increments do not wait on each other
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        for (int j = 0; j < 4; j++) {
            uint32_t b;
            std::memcpy(&b, score + i + j, 4);
            b = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
            u[i + j] = b;
            hist[j][b >> 21]++;
        }
    }
    for (; i < n; i++) {
        uint32_t b;
        std::memcpy(&b, score + i, 4);
        b = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
        u[i] = b;
        hist[0][b >> 21]++;
    }
    int bin = 2047, above = 0;
    auto count = [&](int b) { return static_cast<int>(hist[0][b] + hist[1][b] + hist[2][b] + hist[3][b]); };
    while (above + count(bin) < k) above += count(bin--);
    // branchless compaction: every pool above the bin into out, the bin's into cand (each write always made, the
    // index advanced by the comparison)
    int m = 0, c = 0;
    const uint32_t lo = static_cast<uint32_t>(bin) << 21, hi = lo + (1u << 21);
    for (int j = 0; j < n; j++) {
        const uint32_t v = u[j];
        out[m < k ? m : k - 1] = j;
        m += v >= hi;
        cand[c] = (static_cast<uint64_t>(v) << 32) | static_cast<uint32_t>(~static_cast<uint32_t>(j));
        c += (v >= lo) & (v < hi);
    }
    const int need = k - m;
    std::nth_element(cand.begin(), cand.begin() + (need - 1), cand.begin() + c, std::greater<uint64_t>());
    std::sort(cand.begin(), cand.begin() + need, std::greater<uint64_t>());
    for (int j = 0; j < need; j++) out[m + j] = ~static_cast<uint32_t>(cand[j]);
}
