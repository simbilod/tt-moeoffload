// routing-census: count which routed experts each layer picks, over a corpus of rendered prompts (tools/llama/corpus.py).
//
//   routing-census <llama.cpp common args: -m, -ngl, -ot, -c, -t, -ub ...> \
//       --census-in prompts.jsonl --census-out census.json [--census-decode N] [--census-trace routes.trace]
//       [--census-ids DIR]
//
// prompts.jsonl: one {"id": ..., "prompt": "<fully templated text>"} per line (corpus.py renders the chats with the
// API server's own chat template, so the special tokens are the model's own). Each prompt is processed from an empty cache
// (prompt-phase routing), then N tokens are generated greedily (decode-phase routing, counted separately: a
// decode step routes one token, a prompt chunk thousands - their hot sets need not match).
//
// The eval callback asks for every "ffn_moe_topk-<layer>" tensor (I32 [n_expert_used, n_tokens], the experts the
// router picked per token) and adds them into per-layer histograms. Nothing else is read back.
//
// --census-trace FILE (the lane's warm table, tt/cpulane transitions()): every routing decision in the order the model
// made it, little-endian: per prompt  'P' u32 len, id bytes, u32 prompt tokens;  per callback  'R' u8 phase
// (0 prompt, 1 decode), u16 layer, u32 tokens, u8 k, tokens * k u16 expert ids (token-major).
//
// --census-ids DIR: each prompt's token ids as DIR/<id>.i32 (int32; the checks' prompts, tokenized as the census reads them).
#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"

#include "args.h"

#include <clocale>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"

struct census {
    bool decode_phase = false;
    int n_expert = 0;
    std::map<int, std::vector<uint64_t>> prompt, decode;  // layer -> per-expert route counts
    uint64_t prompt_tokens = 0, decode_tokens = 0;
    std::vector<int32_t> buf;
    FILE * trace = nullptr;
    std::vector<uint16_t> tbuf;
};

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    static const char prefix[] = "ffn_moe_topk-";
    if (strncmp(t->name, prefix, sizeof(prefix) - 1) != 0) {
        return ask ? false : true;
    }
    if (ask) {
        return true;
    }
    auto * c = (census *) user_data;
    const int layer = atoi(t->name + sizeof(prefix) - 1);
    // ffn_moe_topk is a view of the per-token argsort over all experts: ne0 = n_expert_used columns, but rows are
    // nb1 bytes apart (n_expert entries). Read the strided span and take each row's first ne0 entries - reading
    // ggml_nelements() contiguous ints would return whole sorted permutations for any batch above one token.
    const int64_t k = t->ne[0], rows = ggml_nrows(t);
    if (rows == 0) {  // qwen4exp's last layer keeps only the output rows: none for a long prompt's middle batches
        return true;
    }
    const size_t row_ints = t->nb[1] / sizeof(int32_t);
    const size_t span = (size_t) (rows - 1) * t->nb[1] + (size_t) k * sizeof(int32_t);
    GGML_ASSERT(t->type == GGML_TYPE_I32 && t->nb[0] == sizeof(int32_t) && ggml_n_dims(t) <= 2);
    c->buf.resize(span / sizeof(int32_t) + 1);
    ggml_backend_tensor_get(t, c->buf.data(), 0, span);
    auto & hist = (c->decode_phase ? c->decode : c->prompt)[layer];
    if (hist.empty()) {
        hist.assign(c->n_expert, 0);
    }
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t j = 0; j < k; ++j) {
            const int32_t e = c->buf[r * row_ints + j];
            if (e >= 0 && e < c->n_expert) {
                hist[e]++;
            }
        }
    }
    if (c->trace) {
        c->tbuf.resize((size_t) (rows * k));
        for (int64_t r = 0; r < rows; ++r)
            for (int64_t j = 0; j < k; ++j) c->tbuf[(size_t) (r * k + j)] = (uint16_t) c->buf[r * row_ints + j];
        const uint8_t tag = 'R', phase = c->decode_phase ? 1 : 0, kk = (uint8_t) k;
        const uint16_t l16 = (uint16_t) layer;
        const uint32_t n = (uint32_t) rows;
        fwrite(&tag, 1, 1, c->trace); fwrite(&phase, 1, 1, c->trace); fwrite(&l16, 2, 1, c->trace);
        fwrite(&n, 4, 1, c->trace); fwrite(&kk, 1, 1, c->trace);
        fwrite(c->tbuf.data(), 2, c->tbuf.size(), c->trace);
    }
    return true;
}

static void write_hist(FILE * f, const std::map<int, std::vector<uint64_t>> & h) {
    fprintf(f, "{");
    bool first = true;
    for (const auto & [layer, v] : h) {
        fprintf(f, "%s\"%d\": [", first ? "" : ", ", layer);
        for (size_t i = 0; i < v.size(); ++i) {
            fprintf(f, "%s%llu", i ? "," : "", (unsigned long long) v[i]);
        }
        fprintf(f, "]");
        first = false;
    }
    fprintf(f, "}");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    const std::string in_path = take_arg(argc, argv, "--census-in", "");
    const std::string out_path = take_arg(argc, argv, "--census-out", "census.json");
    const int n_decode = std::stoi(take_arg(argc, argv, "--census-decode", "64"));
    const std::string trace_path = take_arg(argc, argv, "--census-trace", "");
    const std::string ids_dir = take_arg(argc, argv, "--census-ids", "");
    if (in_path.empty()) {
        fprintf(stderr, "usage: %s <llama args> --census-in prompts.jsonl --census-out census.json [--census-decode N]\n", argv[0]);
        return 2;
    }

    census c;
    if (!trace_path.empty() && !(c.trace = fopen(trace_path.c_str(), "wb"))) {
        fprintf(stderr, "cannot write %s\n", trace_path.c_str());
        return 1;
    }
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    llama_backend_init();
    params.cb_eval = cb_eval;
    params.cb_eval_user_data = &c;
    params.warmup = false;
    params.sampling.temp = 0.0f;

    auto init = common_init_from_params(params);
    llama_model * model = init->model();
    llama_context * ctx = init->context();
    if (!model || !ctx) {
        LOG_ERR("failed to init\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    char arch[64] = {0}, key[128];
    llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
    char val[32] = {0};
    snprintf(key, sizeof(key), "%s.expert_count", arch);
    llama_model_meta_val_str(model, key, val, sizeof(val));
    c.n_expert = atoi(val);
    if (c.n_expert <= 0) {
        LOG_ERR("no %s in the model metadata\n", key);
        return 1;
    }
    const int n_batch = (int) llama_n_batch(ctx);
    const int n_ctx = (int) llama_n_ctx(ctx);

    std::ifstream in(in_path);
    std::string line;
    int n_prompts = 0;
    while (std::getline(in, line)) {
        const std::string id = json_str(line, "id"), prompt = json_str(line, "prompt");
        if (prompt.empty()) {
            continue;
        }
        std::vector<llama_token> toks = common_tokenize(ctx, prompt, false, true);
        if ((int) toks.size() + n_decode > n_ctx) {
            LOG_WRN("%s: %zu tokens + %d decode > n_ctx %d, skipped\n", id.c_str(), toks.size(), n_decode, n_ctx);
            continue;
        }
        if (!ids_dir.empty()) {
            FILE * fi = fopen((ids_dir + "/" + id + ".i32").c_str(), "wb");
            if (!fi || fwrite(toks.data(), sizeof(int32_t), toks.size(), fi) != toks.size() || fclose(fi) != 0) {
                LOG_ERR("cannot write %s/%s.i32\n", ids_dir.c_str(), id.c_str());
                return 1;
            }
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        c.decode_phase = false;
        if (c.trace) {
            const uint8_t tag = 'P';
            const uint32_t len = (uint32_t) id.size(), nt = (uint32_t) toks.size();
            fwrite(&tag, 1, 1, c.trace); fwrite(&len, 4, 1, c.trace); fwrite(id.data(), 1, len, c.trace);
            fwrite(&nt, 4, 1, c.trace);
        }
        for (size_t off = 0; off < toks.size(); off += n_batch) {
            const int n = (int) std::min<size_t>(n_batch, toks.size() - off);
            common_batch batch = common_batch_get_one(ctx, toks.data() + off, n);
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
                LOG_ERR("%s: decode failed\n", id.c_str());
                return 1;
            }
        }
        c.prompt_tokens += toks.size();
        c.decode_phase = true;
        common_sampler * smpl = common_sampler_init(model, params.sampling);
        for (int i = 0; i < n_decode; ++i) {
            const llama_token t = common_sampler_sample(smpl, ctx, -1);
            common_sampler_accept(smpl, t, true);
            if (llama_vocab_is_eog(vocab, t)) {
                break;
            }
            common_batch batch = common_batch_get_one(ctx, &t, 1);
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
                LOG_ERR("%s: decode step failed\n", id.c_str());
                return 1;
            }
            c.decode_tokens++;
        }
        common_sampler_free(smpl);
        n_prompts++;
        LOG_INF("census: %s done (%zu prompt tokens), totals: %llu prompt / %llu decode\n", id.c_str(), toks.size(),
                (unsigned long long) c.prompt_tokens, (unsigned long long) c.decode_tokens);
    }

    FILE * f = fopen(out_path.c_str(), "w");
    if (!f) {
        LOG_ERR("cannot write %s\n", out_path.c_str());
        return 1;
    }
    fprintf(f, "{\"format_rev\": 1, \"arch\": \"%s\", \"n_expert\": %d, \"prompts\": %d, \"prompt_tokens\": %llu, "
               "\"decode_tokens\": %llu, \"prompt\": ", arch, c.n_expert, n_prompts,
            (unsigned long long) c.prompt_tokens, (unsigned long long) c.decode_tokens);
    write_hist(f, c.prompt);
    fprintf(f, ", \"decode\": ");
    write_hist(f, c.decode);
    fprintf(f, "}\n");
    fclose(f);
    if (c.trace) fclose(c.trace);
    llama_backend_free();
    return 0;
}
