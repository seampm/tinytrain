// Train a Llama-arch transformer from scratch.
//
// Usage:
//   train --tokens <tokens.bin> --out <dir> [--steps N] [--batch-size B]
//         [--seq-len T] [--lr X] [--warmup N] [--weight-decay X] [--seed S]
//         [--ckpt-every N] [--dim D] [--layers L] [--heads H] [--ffn F]
//         [--max-seq M] [--dropout P]
//
// Logs step/loss/tokens-per-second to stdout and <out>/losses.txt;
// checkpoints to <out>/ckpt-<step>/ (tinyinfer-compatible).
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "tinytrain/model.h"
#include "tinytrain/ops.h"
#include "tinytrain/optim.h"

namespace {

struct Args {
    std::string tokens;
    std::string out = "checkpoints";
    std::string resume;      // checkpoint dir to resume weights from
    int64_t start_step = 0;  // global step counter (for LR schedule when resuming)
    int64_t schedule_steps = 0; // LR schedule horizon (0 = start_step + steps)
    int64_t steps = 1000;
    int64_t batch_size = 8;
    int64_t seq_len = 128;
    float lr = 3e-4f;
    int64_t warmup = 100;
    float weight_decay = 0.01f;
    uint64_t seed = 42;
    int64_t ckpt_every = 500;
    int64_t dim = 256, layers = 6, heads = 8, ffn = 768, max_seq = 256;
    float dropout = 0.0f;
};

bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](std::string& v) {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << k << "\n";
                return false;
            }
            v = argv[++i];
            return true;
        };
        std::string v;
        if (k == "--tokens") {
            if (!need(v))
                return false;
            a.tokens = v;
        } else if (k == "--out") {
            if (!need(v))
                return false;
            a.out = v;
        } else if (k == "--steps") {
            if (!need(v))
                return false;
            a.steps = std::stoll(v);
        } else if (k == "--batch-size") {
            if (!need(v))
                return false;
            a.batch_size = std::stoll(v);
        } else if (k == "--seq-len") {
            if (!need(v))
                return false;
            a.seq_len = std::stoll(v);
        } else if (k == "--lr") {
            if (!need(v))
                return false;
            a.lr = std::stof(v);
        } else if (k == "--warmup") {
            if (!need(v))
                return false;
            a.warmup = std::stoll(v);
        } else if (k == "--weight-decay") {
            if (!need(v))
                return false;
            a.weight_decay = std::stof(v);
        } else if (k == "--seed") {
            if (!need(v))
                return false;
            a.seed = std::stoull(v);
        } else if (k == "--ckpt-every") {
            if (!need(v))
                return false;
            a.ckpt_every = std::stoll(v);
        } else if (k == "--dim") {
            if (!need(v))
                return false;
            a.dim = std::stoll(v);
        } else if (k == "--layers") {
            if (!need(v))
                return false;
            a.layers = std::stoll(v);
        } else if (k == "--heads") {
            if (!need(v))
                return false;
            a.heads = std::stoll(v);
        } else if (k == "--ffn") {
            if (!need(v))
                return false;
            a.ffn = std::stoll(v);
        } else if (k == "--max-seq") {
            if (!need(v))
                return false;
            a.max_seq = std::stoll(v);
        } else if (k == "--dropout") {
            if (!need(v))
                return false;
            a.dropout = std::stof(v);
        } else if (k == "--resume") {
            if (!need(v))
                return false;
            a.resume = v;
        } else if (k == "--start-step") {
            if (!need(v))
                return false;
            a.start_step = std::stoll(v);
        } else if (k == "--schedule-steps") {
            if (!need(v))
                return false;
            a.schedule_steps = std::stoll(v);
        } else {
            std::cerr << "unknown arg: " << k << "\n";
            return false;
        }
    }
    if (a.tokens.empty()) {
        std::cerr << "--tokens is required\n";
        return false;
    }
    return true;
}

// deterministic xorshift64* (same family as the engine RNG)
struct Rng {
    explicit Rng(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    uint64_t next() {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return s * 0x2545F4914F6CDD1DULL;
    }
    uint64_t s;
};

} // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse(argc, argv, a))
        return 1;

    // load token stream
    std::ifstream f(a.tokens, std::ios::binary | std::ios::ate);
    if (!f) {
        std::cerr << "cannot open " << a.tokens << "\n";
        return 1;
    }
    int64_t nbytes = f.tellg();
    f.seekg(0);
    int64_t ntok = nbytes / 4;
    std::vector<int32_t> tokens(static_cast<size_t>(ntok));
    f.read(reinterpret_cast<char*>(tokens.data()), nbytes);
    std::cout << "tokens: " << ntok << "\n";

    tt::GPTConfig cfg;
    cfg.dim = a.dim;
    cfg.n_layers = a.layers;
    cfg.n_heads = a.heads;
    cfg.n_kv_heads = a.heads;
    cfg.ffn_dim = a.ffn;
    cfg.max_seq_len = a.max_seq;
    cfg.dropout = a.dropout;
    // vocab_size stays 32000 (llama tokenizer)

    tt::GPTModel model(cfg, a.seed);
    if (!a.resume.empty()) {
        model.load_checkpoint(a.resume);
        std::cout << "resumed weights from " << a.resume << "\n";
    }
    int64_t nparams = 0;
    for (auto* p : model.parameters())
        nparams += p->numel();
    std::cout << "parameters: " << nparams << "\n";

    tt::AdamW opt(model.parameters(), a.lr, 0.9f, 0.999f, 1e-8f, a.weight_decay);

    Rng rng(a.seed + 999);
    std::string mk = "mkdir -p " + a.out;
    if (system(mk.c_str()) != 0) {
        std::cerr << "cannot create " << a.out << "\n";
        return 1;
    }
    std::ofstream logf(a.out + "/losses.txt", a.start_step > 0 ? std::ios::app : std::ios::trunc);

    int64_t B = a.batch_size, T = a.seq_len;
    std::vector<int64_t> idx(static_cast<size_t>(B * T));
    std::vector<int64_t> targets(static_cast<size_t>(B * T));

    int64_t total_steps = a.schedule_steps > 0 ? a.schedule_steps : a.start_step + a.steps;
    auto t0 = std::chrono::steady_clock::now();
    for (int64_t step = 0; step < a.steps; ++step) {
        int64_t gstep = a.start_step + step; // global step for the LR schedule
        // cosine schedule with linear warmup, decaying to 10% of peak
        float lr_scale;
        if (gstep < a.warmup) {
            lr_scale = static_cast<float>(gstep + 1) / static_cast<float>(a.warmup);
        } else {
            float prog = static_cast<float>(gstep - a.warmup) /
                         static_cast<float>(std::max<int64_t>(total_steps - a.warmup, 1));
            lr_scale = 0.1f + 0.9f * 0.5f * (1.0f + std::cos(3.14159265f * prog));
        }

        // sample B random length-(T+1) windows
        for (int64_t b = 0; b < B; ++b) {
            int64_t start = static_cast<int64_t>(rng.next() % (ntok - T - 1));
            for (int64_t t = 0; t < T; ++t) {
                idx[static_cast<size_t>(b * T + t)] = tokens[static_cast<size_t>(start + t)];
                targets[static_cast<size_t>(b * T + t)] =
                    tokens[static_cast<size_t>(start + t + 1)];
            }
        }

        opt.zero_grad();
        tt::Tensor logits = model.forward(idx, B, T);
        tt::Tensor loss = tt::cross_entropy(logits, targets);
        float lossv = loss.data()[0];
        loss.backward();
        opt.set_lr(a.lr * lr_scale);
        opt.step();

        auto t1 = std::chrono::steady_clock::now();
        double secs =
            std::chrono::duration<double>(t1 - t0).count() / (step + 1);
        double tps = (B * T) / secs;
        std::cout << "step " << gstep << " loss " << lossv << " lr_scale "
                  << lr_scale << " tok/s " << static_cast<int64_t>(tps) << "\n";
        logf << gstep << " " << lossv << "\n";
        logf.flush();

        if ((gstep + 1) % a.ckpt_every == 0) {
            std::string dir = a.out + "/ckpt-" + std::to_string(gstep + 1);
            model.save_checkpoint(dir);
            std::cout << "checkpoint: " << dir << "\n";
        }
    }
    model.save_checkpoint(a.out + "/ckpt-final");
    std::cout << "done\n";
    return 0;
}
