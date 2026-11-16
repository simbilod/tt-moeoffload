// The token's inputs on the host (G4): its embedding row and its PLE rows, dequantized by ggml's own to_float
// (ggml_get_type_traits), and step_traced's input buffers filled from them (g4_model.Inputs.fill). The arithmetic is
// g4_host.Host's (embed, ple_rows, ple) and Inputs.fill's torch path, to the bit (checked by
// tt/flashnext/g4_host_loop.py):
//
//   gt_embed      token_embd's row tok, 4 copies (X's init: the hyper-connection streams, fp32 [4 x 2560])
//   gt_ple        the n-gram hash of tok and its predecessors into per_layer_token_embd's rows, gathered: fp32 [2560]
//   gt_fill       X fp32 and bf16 (round to nearest even, as torch's), the PLE input bf16, the step words' position
//
// token_embd (Q8_0, 675 MB) is read into memory once; per_layer_token_embd (IQ4_NL, 28.8 GB) is mapped from the file,
// as gguf-py's memmap: its rows come from the page cache.
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "ggml.h"

namespace {

constexpr int D = 2560, HC = 4, MAXGRAM = 8, MAXHEADS = 64;

struct Table {
    const uint8_t* data = nullptr;  // row 0
    size_t row_bytes = 0;
    int64_t rows = 0, ne0 = 0;
    ggml_to_float_t to_float = nullptr;
} embd, ple;

int n_gram = 0, per_gram = 0, n_heads = 0;
int64_t eos = 0;
uint64_t mult[MAXGRAM], offsets[MAXHEADS], sizes[MAXHEADS];

int open_table(Table& t, const char* path, uint64_t offset, int type, int64_t rows, int64_t ne0, bool read_in) {
    const ggml_type_traits* tr = ggml_get_type_traits((ggml_type) type);
    if (!tr || !tr->to_float || ne0 % tr->blck_size) return -22;
    const size_t row_bytes = ggml_row_size((ggml_type) type, ne0), bytes = row_bytes * rows;
    const int fd = open(path, O_RDONLY);
    if (fd < 0) return -2;
    const uint8_t* data;
    if (read_in) {
        void* m = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m == MAP_FAILED) return close(fd), -12;
        for (size_t done = 0; done < bytes;) {
            const ssize_t r = pread(fd, (uint8_t*) m + done, bytes - done > (64u << 20) ? (64u << 20) : bytes - done,
                                    offset + done);
            if (r <= 0) return munmap(m, bytes), close(fd), -5;
            done += r;
        }
        data = (const uint8_t*) m;
    } else {
        const uint64_t base = offset & ~(uint64_t) (sysconf(_SC_PAGESIZE) - 1);
        void* m = mmap(nullptr, bytes + (offset - base), PROT_READ, MAP_SHARED, fd, base);
        if (m == MAP_FAILED) return close(fd), -12;
        madvise(m, bytes + (offset - base), MADV_RANDOM);
        data = (const uint8_t*) m + (offset - base);
    }
    close(fd);
    t = {data, row_bytes, rows, ne0, tr->to_float};
    return 0;
}

// torch's float -> bfloat16 as its vectorized copy_ does it (ATen's cvtfp32_bf16: round to nearest even, every NaN
// 0xFFFF; c10's scalar conversion gives 0x7FC0)
inline uint16_t bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if (f != f) return 0xFFFF;
    return (uint16_t) ((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

}  // namespace

extern "C" {

// token_embd: rows x ne0 of ggml type `type` at `offset` in `path`, read into memory. Returns 0 or -errno.
int gt_open_embd(const char* path, uint64_t offset, int type, int64_t rows, int64_t ne0) {
    if (ne0 != D) return -22;
    return embd.data ? 0 : open_table(embd, path, offset, type, rows, ne0, true);
}

// per_layer_token_embd (mapped) and the PLE hash's parameters (g4_host.Host: qwen4exp.ple.*): n-gram size, heads per
// n-gram, the EOS id, the layer multipliers [n_gram], the heads' offsets and vocab sizes [(n_gram - 1) per_gram].
int gt_open_ple(const char* path, uint64_t offset, int type, int64_t rows, int64_t ne0, int ngram, int pergram,
                int64_t eos_id, const uint64_t* mults, const uint64_t* offs, const uint64_t* vocab) {
    const int heads = (ngram - 1) * pergram;
    if (ngram < 2 || ngram > MAXGRAM || heads > MAXHEADS || heads * ne0 != D) return -22;
    for (int h = 0; h < heads; h++)
        if (!vocab[h] || offs[h] + vocab[h] > (uint64_t) rows) return -34;
    if (!ple.data) {
        const int rc = open_table(ple, path, offset, type, rows, ne0, false);
        if (rc) return rc;
    }
    n_gram = ngram, per_gram = pergram, n_heads = heads, eos = eos_id;
    std::memcpy(mult, mults, ngram * sizeof(uint64_t));
    std::memcpy(offsets, offs, heads * sizeof(uint64_t));
    std::memcpy(sizes, vocab, heads * sizeof(uint64_t));
    return 0;
}

// X's init for token tok: fp32 [HC x D]. Returns 0, or -1 for a token out of range.
int gt_embed(int64_t tok, float* out) {
    if (tok < 0 || tok >= embd.rows) return -1;
    embd.to_float(embd.data + (size_t) tok * embd.row_bytes, out, D);
    for (int s = 1; s < HC; s++) std::memcpy(out + s * D, out, D * sizeof(float));
    return 0;
}

// The PLE rows of tok (Host.ple_rows): prev its last n_prev predecessors, oldest first. idx: int64 [n_heads].
void gt_ple_rows(int64_t tok, const int64_t* prev, int n_prev, int64_t* idx) {
    uint64_t ctx[MAXGRAM] = {(uint64_t) tok};
    bool cut = false;
    for (int s = 1; s < n_gram; s++) {  // s positions back; none there reads as EOS, as does all before one
        const int64_t t = s <= n_prev ? prev[n_prev - s] : -1;
        cut = cut || t < 0 || t == eos;
        ctx[s] = (uint64_t) (cut ? eos : t);
    }
    for (int n = 2; n <= n_gram; n++) {
        uint64_t mixed = ctx[0] * mult[0];
        for (int j = 1; j < n; j++) mixed ^= ctx[j] * mult[j];
        for (int g = 0; g < per_gram; g++) {
            const int h = (n - 2) * per_gram + g;
            idx[h] = (int64_t) (mixed % sizes[h] + offsets[h]);
        }
    }
}

// The PLE input of tok (Host.ple): fp32 [D], head-major. Returns 0, or -1 for a token out of range.
int gt_ple(int64_t tok, const int64_t* prev, int n_prev, float* out) {
    if (tok < 0) return -1;
    int64_t idx[MAXHEADS];
    gt_ple_rows(tok, prev, n_prev, idx);
    for (int h = 0; h < n_heads; h++)
        ple.to_float(ple.data + (size_t) idx[h] * ple.row_bytes, out + h * ple.ne0, ple.ne0);
    return 0;
}

// A prefill chunk's inputs (g7_prefill.token_inputs): for its tokens p0 .. p0 + n - 1 of ids, X's init (one stream of
// gt_embed's) into x0 [n x D] and gt_ple's rows (n_prev predecessors at most) into emb [n x D], on `threads` threads
// (from Python a loop of the two calls a token was GIL-bound: 0.49 s at 8,192 tokens). Returns 0, or -1 for a token
// out of range.
int gt_inputs(const int64_t* ids, int64_t p0, int n, int n_prev, float* x0, float* emb, int threads) {
    std::vector<int> bad(threads, 0);
    std::vector<std::thread> pool;
    for (int k = 0; k < threads; k++)
        pool.emplace_back([&, k] {
            std::vector<float> x4(HC * D);
            for (int i = k; i < n; i += threads) {
                const int64_t q = p0 + i, s = q > n_prev ? q - n_prev : 0;
                bad[k] |= gt_embed(ids[q], x4.data());
                std::memcpy(x0 + (size_t) i * D, x4.data(), D * sizeof(float));
                bad[k] |= gt_ple(ids[q], ids + s, (int) (q - s), emb + (size_t) i * D);
            }
        });
    int any = 0;
    for (int k = 0; k < threads; k++) {
        pool[k].join();
        any |= bad[k];
    }
    return any ? -1 : 0;
}

// Inputs.fill: x0 fp32 [HC x D] into x32 and (bf16) x16, emb fp32 [D] into emb16 (bf16), pos into step[0].
void gt_fill(const float* x0, const float* emb, int32_t pos, float* x32, uint16_t* x16, uint16_t* emb16,
             int32_t* step) {
    std::memcpy(x32, x0, HC * D * sizeof(float));
    for (int i = 0; i < HC * D; i++) x16[i] = bf16(x0[i]);
    for (int i = 0; i < D; i++) emb16[i] = bf16(emb[i]);
    step[0] = pos;
}

}  // extern "C"
