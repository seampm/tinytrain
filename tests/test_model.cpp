// Optimizer convergence, model shapes, and an end-to-end gradient check
// through a tiny transformer (attention, RoPE, SwiGLU, tied head).
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "tinytrain/model.h"
#include "tinytrain/nn.h"
#include "tinytrain/ops.h"
#include "tinytrain/optim.h"

TEST(Optim, AdamWMinimizesQuadratic) {
    // min ||x - target||^2 ; AdamW should drive x to target.
    tt::Tensor x = tt::full({4}, 0.0f, true);
    std::vector<float> target = {1.0f, -2.0f, 3.0f, -4.0f};
    tt::AdamW opt({&x}, 0.1f);
    for (int i = 0; i < 500; ++i) {
        opt.zero_grad();
        tt::Tensor diff = tt::sub(x, tt::full({4}, 0.0f));
        // manual target shift: loss = sum((x - t)^2)
        tt::Tensor t = tt::full({4}, 0.0f);
        for (int j = 0; j < 4; ++j)
            t.data()[j] = target[j];
        tt::Tensor d = tt::sub(x, t);
        tt::Tensor loss = tt::sum_all(tt::mul(d, d));
        loss.backward();
        opt.step();
    }
    for (int j = 0; j < 4; ++j)
        EXPECT_NEAR(x.data()[j], target[j], 1e-3) << "j=" << j;
}

TEST(Optim, SGDMomentumMinimizesQuadratic) {
    tt::Tensor x = tt::full({3}, 5.0f, true);
    tt::SGD opt({&x}, 0.05f, 0.9f);
    for (int i = 0; i < 300; ++i) {
        opt.zero_grad();
        tt::Tensor loss = tt::sum_all(tt::mul(x, x)); // min ||x||^2
        loss.backward();
        opt.step();
    }
    for (int j = 0; j < 3; ++j)
        EXPECT_NEAR(x.data()[j], 0.0, 1e-3) << "j=" << j;
}

namespace {

tt::GPTConfig tiny_cfg() {
    tt::GPTConfig c;
    c.vocab_size = 16;
    c.dim = 8;
    c.n_layers = 1;
    c.n_heads = 2;
    c.n_kv_heads = 2;
    c.ffn_dim = 16;
    c.max_seq_len = 8;
    return c;
}

} // namespace

TEST(Model, ForwardShapes) {
    tt::GPTModel m(tiny_cfg(), 7);
    std::vector<int64_t> idx = {1, 2, 3, 4, 5, 6, 7, 8}; // B=2, T=4
    tt::Tensor logits = m.forward(idx, 2, 4);
    ASSERT_EQ(logits.ndim(), 2);
    EXPECT_EQ(logits.dim(0), 8);
    EXPECT_EQ(logits.dim(1), 16);
    // logits must be finite
    for (int64_t i = 0; i < logits.numel(); ++i)
        EXPECT_TRUE(std::isfinite(logits.data()[i]));
}

TEST(Model, GradcheckThroughTransformer) {
    // The whole model, differentiated end to end vs finite differences.
    tt::GPTModel m(tiny_cfg(), 11);
    std::vector<int64_t> idx = {1, 2, 3, 4};
    std::vector<int64_t> targets = {2, 3, 4, 5};
    auto fwd = [&]() { return tt::cross_entropy(m.forward(idx, 1, 4), targets); };

    auto params = m.parameters();
    for (auto* p : params)
        p->zero_grad();
    fwd().backward();

    const double eps = 1e-3;
    // check a subset of parameters (full check is slow); cover every kind:
    // embedding table, a q-proj, an rmsnorm weight.
    std::vector<tt::Tensor*> subset = {params[0], params[2], params[1]};
    for (tt::Tensor* p : subset) {
        ASSERT_TRUE(p->has_grad());
        int64_t check = std::min<int64_t>(p->numel(), 12);
        for (int64_t i = 0; i < check; ++i) {
            float orig = p->data()[i];
            p->data()[i] = orig + eps;
            double lp = fwd().data()[0];
            p->data()[i] = orig - eps;
            double lm = fwd().data()[0];
            p->data()[i] = orig;
            double numeric = (lp - lm) / (2 * eps);
            double analytic = p->grad_data()[i];
            double tol = 2e-3 + 2e-2 * std::fabs(numeric);
            EXPECT_LE(std::fabs(analytic - numeric), tol)
                << "param idx " << i << ": analytic=" << analytic
                << " numeric=" << numeric;
        }
    }
}

TEST(Model, CheckpointRoundtrip) {
    tt::GPTModel m(tiny_cfg(), 13);
    std::string dir = "/tmp/tinytrain_ckpt_test";
    std::filesystem::remove_all(dir);
    m.save_checkpoint(dir);

    // config.json parses and carries the architecture
    std::ifstream cf(dir + "/config.json");
    ASSERT_TRUE(cf.good());
    nlohmann::json cfg;
    cf >> cfg;
    EXPECT_EQ(cfg["hidden_size"], 8);
    EXPECT_EQ(cfg["num_hidden_layers"], 1);
    EXPECT_EQ(cfg["vocab_size"], 16);

    // model.safetensors: header parses, offsets are contiguous, payload size matches
    std::ifstream f(dir + "/model.safetensors", std::ios::binary);
    ASSERT_TRUE(f.good());
    uint64_t hlen = 0;
    f.read(reinterpret_cast<char*>(&hlen), 8);
    std::string hstr(hlen, '\0');
    f.read(hstr.data(), static_cast<std::streamsize>(hlen));
    nlohmann::json header = nlohmann::json::parse(hstr);
    EXPECT_TRUE(header.contains("model.embed_tokens.weight"));
    EXPECT_TRUE(header.contains("model.layers.0.self_attn.q_proj.weight"));
    EXPECT_TRUE(header.contains("model.norm.weight"));
    EXPECT_FALSE(header.contains("lm_head.weight")); // tied
    int64_t total = 0;
    for (auto& [name, entry] : header.items()) {
        EXPECT_EQ(entry["dtype"], "F32");
        int64_t n = 1;
        for (auto d : entry["shape"])
            n *= d.get<int64_t>();
        EXPECT_EQ(entry["data_offsets"][1].get<int64_t>() -
                      entry["data_offsets"][0].get<int64_t>(),
                  n * 4);
        total += n * 4;
    }
    f.seekg(0, std::ios::end);
    EXPECT_EQ(static_cast<int64_t>(f.tellg()), static_cast<int64_t>(8 + hlen) + total);
    std::filesystem::remove_all(dir);
}
