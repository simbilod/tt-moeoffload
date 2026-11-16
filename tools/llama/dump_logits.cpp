// dump-logits: the logits of every position of a fixed token-id sequence, as f32 [n_tokens, n_vocab], position-major.
//
//   dump-logits <llama.cpp common args: -m, -ngl, -ot, -c, -t, -ub ...> --dump-ids ids.i32 --dump-out out.f32
//                     [--dump-tensors ffn_norm,ffn_moe_out --dump-dir DIR] [--dump-from N]
//
// --dump-from N: only positions >= N get logits (a long context's last positions: [n_tokens - N, n_vocab]).
//
// --dump-tensors: every layer's named graph tensors (llama.cpp's cb names, e.g. ffn_norm-12) as f32 [n_tokens, n_embd]
// files DIR/<name>-<layer>.f32 (a graph-level tensor, e.g. model.input_embed, as DIR/<name>.f32), for a per-layer bisect against Strata's STRATA_LAYER_DUMP (L??_fn / L??_routed).
//
// Built against the pinned third_party/llama.cpp by tools/llama/build.sh.
#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#include "ggml-backend.h"
#include "ggml.h"

#include "args.h"

#include <clocale>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

struct dump_state {
    std::vector<std::string> names;
    std::string dir;
    std::vector<float> buf;
};

static bool dump_cb(struct ggml_tensor * t, bool ask, void * ud) {
    auto * d = (dump_state *) ud;
    const std::string n = t->name;
    bool want = false;
    for (const auto & w : d->names) {
        // a layer's tensor is named w-<il>; the graph's own (il -1: model.input_embed, result_norm) w alone
        want = want || n == w ||
               (n.rfind(w + "-", 0) == 0 && n.find_first_not_of("0123456789", w.size() + 1) == std::string::npos);
    }
    if (ask || !want) {
        return ask ? want : true;
    }
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
        fprintf(stderr, "dump: %s is not contiguous f32, skipped\n", t->name);
        return true;
    }
    d->buf.resize((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, d->buf.data(), 0, ggml_nbytes(t));
    // a multi-ubatch prompt appends: the rows come in position order
    std::FILE * f = std::fopen((d->dir + "/" + n + ".f32").c_str(), "ab");
    std::fwrite(d->buf.data(), sizeof(float), d->buf.size(), f);
    std::fclose(f);
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    const std::string ids_path = take_arg(argc, argv, "--dump-ids"), out_path = take_arg(argc, argv, "--dump-out");
    const std::string tensors = take_arg(argc, argv, "--dump-tensors");
    const std::string from_arg = take_arg(argc, argv, "--dump-from");
    const size_t from = from_arg.empty() ? 0 : (size_t) std::stoul(from_arg);
    dump_state ds;
    ds.dir = take_arg(argc, argv, "--dump-dir");
    for (size_t a = 0, b; a < tensors.size(); a = b + 1) {
        b = tensors.find(',', a);
        if (b == std::string::npos) b = tensors.size();
        ds.names.push_back(tensors.substr(a, b - a));
    }
    if (ids_path.empty() || out_path.empty()) {
        fprintf(stderr, "usage: %s <llama args> --dump-ids ids.i32 --dump-out out.f32\n", argv[0]);
        return 2;
    }
    std::ifstream in(ids_path, std::ios::binary | std::ios::ate);
    std::vector<llama_token> ids((size_t) in.tellg() / sizeof(int32_t));
    in.seekg(0);
    in.read((char *) ids.data(), (std::streamsize) (ids.size() * sizeof(int32_t)));

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    llama_backend_init();
    params.warmup = false;
    if (!ds.names.empty()) {
        params.cb_eval = dump_cb;
        params.cb_eval_user_data = &ds;
    }
    auto init = common_init_from_params(params);
    llama_context * ctx = init->context();
    if (!init->model() || !ctx) {
        LOG_ERR("failed to init\n");
        return 1;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(init->model()));
    const int n_batch = (int) llama_n_batch(ctx);
    std::FILE * f = std::fopen(out_path.c_str(), "wb");
    if (!f) {
        LOG_ERR("cannot write %s\n", out_path.c_str());
        return 1;
    }
    for (size_t off = 0; off < ids.size(); off += (size_t) n_batch) {
        const int n = (int) std::min<size_t>((size_t) n_batch, ids.size() - off);
        common_batch batch(ctx);
        for (int i = 0; i < n; ++i) {
            batch.add(ids[off + (size_t) i], (llama_pos) (off + (size_t) i), 0, off + (size_t) i >= from);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            LOG_ERR("decode failed at %zu\n", off);
            return 1;
        }
        for (int i = 0; i < n; ++i) {
            if (off + (size_t) i >= from) {
                std::fwrite(llama_get_logits_ith(ctx, i), sizeof(float), (size_t) n_vocab, f);
            }
        }
    }
    std::fclose(f);
    LOG_INF("dumped %zu x %d logits to %s\n", ids.size() - std::min(from, ids.size()), n_vocab, out_path.c_str());
    llama_backend_free();
    return 0;
}
