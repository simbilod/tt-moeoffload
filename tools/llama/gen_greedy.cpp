// gen-greedy: llama.cpp's own greedy continuations of a prompt corpus, with what an MTP draft layer needs from
// the target at every position: the final multi-stream residual (qwen4exp's "l_last-47", 4 x 2560 f32) and the
// target's top-8 logits. The verifier for tt/flashnext/w4_draft.py's acceptance replay.
//
//   gen-greedy <llama.cpp common args: -m, -ngl, -ot, -c, -b, -ub, -t ...> --gen-in prompts.jsonl
//       --gen-out DIR [--gen-decode 256] [--gen-keep 8192] [--gen-ids A.code.0,A.code.1] [--gen-layer l_last-47]
//       [--gen-width 10240]
//
// Per prompt (DIR/<id>/): every prompt row is a logits output (qwen4exp's last layer keeps only the output rows),
// then N greedy tokens are decoded one at a time (llama.cpp's decode path, argmax of the logits) until N or an EOG.
//   ids.i32   int32 [L]               prompt + generated ids (the EOG, if one came, is the last id and is not decoded)
//   R.f32     f32 [rows, 4 x 2560]    the residual streams after the last layer (stream-major), positions r0 ..
//                                     r0 + rows - 1 (rows = L - r0, or L - 1 - r0 after an EOG)
//   top.i32   int32 [rows, 8]         per R row: the target's top-8 next-token ids (row i predicts ids[r0 + i + 1])
//   top.f32   f32 [rows, 9]           their logits, then the row's logsumexp
//   meta.json {n_prompt, r0, n_gen, rows, eog}     r0 = max(0, n_prompt - keep): older prompt rows are not kept
// Built against the pinned third_party/llama.cpp by tools/llama/build.sh.
#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#include "ggml-backend.h"
#include "ggml.h"

#include "args.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <vector>

struct gen_state {
    std::string layer = "l_last-47";
    int64_t pos0 = 0;       // the absolute position of the current ubatch's first row
    int64_t keep_from = 0;  // rows before this position are dropped
    int64_t rows = 0, width = 10240;   // hc x n_embd
    FILE * f = nullptr;
    std::vector<float> buf;
};

static bool cb_eval(struct ggml_tensor * t, bool ask, void * ud) {
    auto * s = (gen_state *) ud;
    if (s->layer != t->name) {
        return ask ? false : true;
    }
    if (ask) {
        return true;
    }
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
        fprintf(stderr, "gen: %s is not contiguous f32\n", t->name);
        return false;
    }
    // [n_embd, hc, n_tokens] (or [n_embd * hc, n_tokens]): rows of `width` floats, in position order
    const int64_t w = s->width, n = ggml_nelements(t) / w;
    if (n == 0 || s->f == nullptr) {
        return true;
    }
    s->buf.resize((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, s->buf.data(), 0, ggml_nbytes(t));
    for (int64_t i = 0; i < n; ++i) {
        if (s->pos0 + i >= s->keep_from) {
            fwrite(s->buf.data() + i * w, sizeof(float), (size_t) w, s->f);
            s->rows++;
        }
    }
    return true;
}

// top-8 ids and logits of one row, and its logsumexp
static void top8(const float * lg, int n_vocab, int32_t * ids, float * vals) {
    std::vector<int32_t> idx(8, -1);
    std::vector<float> v(8, -INFINITY);
    float mx = -INFINITY;
    for (int i = 0; i < n_vocab; ++i) {
        mx = std::max(mx, lg[i]);
        if (lg[i] > v[7]) {
            int j = 7;
            while (j > 0 && lg[i] > v[j - 1]) {
                v[j] = v[j - 1];
                idx[j] = idx[j - 1];
                --j;
            }
            v[j] = lg[i];
            idx[j] = i;
        }
    }
    double s = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        s += std::exp((double) lg[i] - mx);
    }
    for (int j = 0; j < 8; ++j) {
        ids[j] = idx[j];
        vals[j] = v[j];
    }
    vals[8] = (float) (mx + std::log(s));
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    const std::string in_path = take_arg(argc, argv, "--gen-in", "");
    const std::string out_dir = take_arg(argc, argv, "--gen-out", "");
    const int n_decode = std::stoi(take_arg(argc, argv, "--gen-decode", "256"));
    const int64_t keep = std::stoll(take_arg(argc, argv, "--gen-keep", "8192"));
    const std::string only = take_arg(argc, argv, "--gen-ids", "");
    gen_state st;
    st.layer = take_arg(argc, argv, "--gen-layer", "l_last-47");
    st.width = std::stoll(take_arg(argc, argv, "--gen-width", "10240"));
    if (in_path.empty() || out_dir.empty()) {
        fprintf(stderr, "usage: %s <llama args> --gen-in prompts.jsonl --gen-out DIR [--gen-decode N]\n", argv[0]);
        return 2;
    }
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    llama_backend_init();
    params.warmup = false;
    params.cb_eval = cb_eval;
    params.cb_eval_user_data = &st;
    auto init = common_init_from_params(params);
    llama_model * model = init->model();
    llama_context * ctx = init->context();
    if (!model || !ctx) {
        LOG_ERR("failed to init\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_batch = (int) llama_n_batch(ctx), n_ctx = (int) llama_n_ctx(ctx);
    mkdir(out_dir.c_str(), 0775);

    std::ifstream in(in_path);
    std::string line;
    while (std::getline(in, line)) {
        const std::string id = json_str(line, "id"), prompt = json_str(line, "prompt");
        if (prompt.empty() || (!only.empty() && ("," + only + ",").find("," + id + ",") == std::string::npos)) {
            continue;
        }
        std::vector<llama_token> ids = common_tokenize(ctx, prompt, false, true);
        const int64_t n_prompt = (int64_t) ids.size();
        if (n_prompt + n_decode > n_ctx) {
            LOG_WRN("%s: %lld tokens + %d decode > n_ctx %d, skipped\n", id.c_str(), (long long) n_prompt, n_decode, n_ctx);
            continue;
        }
        const std::string dir = out_dir + "/" + id;
        mkdir(dir.c_str(), 0775);
        st.f = std::fopen((dir + "/R.f32").c_str(), "wb");
        FILE * ft = std::fopen((dir + "/top.i32").c_str(), "wb");
        FILE * fv = std::fopen((dir + "/top.f32").c_str(), "wb");
        if (!st.f || !ft || !fv) {
            LOG_ERR("cannot write in %s\n", dir.c_str());
            return 1;
        }
        st.rows = 0;
        st.keep_from = std::max<int64_t>(0, n_prompt - keep);
        llama_memory_clear(llama_get_memory(ctx), true);
        int32_t tid[8];
        float tv[9];
        llama_token x = LLAMA_TOKEN_NULL;
        for (int64_t off = 0; off < n_prompt; off += n_batch) {
            const int n = (int) std::min<int64_t>(n_batch, n_prompt - off);
            common_batch batch(ctx);
            for (int i = 0; i < n; ++i) {
                batch.add(ids[(size_t) (off + i)], (llama_pos) (off + i), 0, true);
            }
            st.pos0 = off;
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
                LOG_ERR("%s: prompt decode failed at %lld\n", id.c_str(), (long long) off);
                return 1;
            }
            for (int i = 0; i < n; ++i) {
                if (off + i < st.keep_from) {
                    continue;
                }
                const float * lg = llama_get_logits_ith(ctx, i);
                top8(lg, n_vocab, tid, tv);
                fwrite(tid, 4, 8, ft);
                fwrite(tv, 4, 9, fv);
                if (off + i == n_prompt - 1) {
                    x = tid[0];
                }
            }
        }
        int n_gen = 0;
        bool eog = false;
        for (int i = 0; i < n_decode; ++i) {
            ids.push_back(x);
            ++n_gen;
            if (llama_vocab_is_eog(vocab, x)) {
                eog = true;
                break;
            }
            common_batch batch(ctx);
            batch.add(x, (llama_pos) (n_prompt + i), 0, true);
            st.pos0 = n_prompt + i;
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
                LOG_ERR("%s: decode step %d failed\n", id.c_str(), i);
                return 1;
            }
            top8(llama_get_logits_ith(ctx, -1), n_vocab, tid, tv);
            fwrite(tid, 4, 8, ft);
            fwrite(tv, 4, 9, fv);
            x = tid[0];
        }
        std::fclose(st.f);
        st.f = nullptr;
        std::fclose(ft);
        std::fclose(fv);
        FILE * fi = std::fopen((dir + "/ids.i32").c_str(), "wb");
        fwrite(ids.data(), sizeof(int32_t), ids.size(), fi);
        std::fclose(fi);
        FILE * fm = std::fopen((dir + "/meta.json").c_str(), "w");
        fprintf(fm, "{\"id\": \"%s\", \"n_prompt\": %lld, \"r0\": %lld, \"n_gen\": %d, \"rows\": %lld, \"width\": %lld, "
                    "\"eog\": %s, \"layer\": \"%s\"}\n", id.c_str(), (long long) n_prompt, (long long) st.keep_from, n_gen,
                (long long) st.rows, (long long) st.width, eog ? "true" : "false", st.layer.c_str());
        std::fclose(fm);
        LOG_INF("gen: %s: %lld prompt + %d generated, %lld rows of %s\n", id.c_str(), (long long) n_prompt, n_gen,
                (long long) st.rows, st.layer.c_str());
    }
    llama_backend_free();
    return 0;
}
