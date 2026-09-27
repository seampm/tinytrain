#include "tinytrain/model.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "tinytrain/ops.h"

namespace tt {

GPTModel::GPTModel(const GPTConfig& cfg, uint64_t seed) : cfg_(cfg) {
    if (cfg.dim % cfg.n_heads != 0)
        throw std::invalid_argument("GPTModel: dim must divide n_heads");
    int64_t hd = cfg.dim / cfg.n_heads;
    if (hd % 2 != 0)
        throw std::invalid_argument("GPTModel: head_dim must be even");

    tok_embeddings = Embedding(cfg.vocab_size, cfg.dim, seed + 1);
    blocks_.reserve(static_cast<size_t>(cfg.n_layers));
    for (int64_t l = 0; l < cfg.n_layers; ++l) {
        Block b;
        uint64_t s = seed + 100 + static_cast<uint64_t>(l) * 10;
        b.rms_attn = RMSNorm(cfg.dim, cfg.norm_eps);
        b.wq = Linear(cfg.dim, cfg.n_heads * hd, s + 0);
        b.wk = Linear(cfg.dim, cfg.n_kv_heads * hd, s + 1);
        b.wv = Linear(cfg.dim, cfg.n_kv_heads * hd, s + 2);
        b.wo = Linear(cfg.n_heads * hd, cfg.dim, s + 3);
        b.rms_ffn = RMSNorm(cfg.dim, cfg.norm_eps);
        b.wgate = Linear(cfg.dim, cfg.ffn_dim, s + 4);
        b.wup = Linear(cfg.dim, cfg.ffn_dim, s + 5);
        b.wdown = Linear(cfg.ffn_dim, cfg.dim, s + 6);
        blocks_.push_back(std::move(b));
    }
    rms_final = RMSNorm(cfg.dim, cfg.norm_eps);
    // Tied embeddings: lm_head reuses the token embedding table.
    // (tinyinfer falls back to the embedding table when lm_head.weight is absent.)
    tie_weights_ = true;

    // RoPE tables: cos/sin of pos / theta^(2i/hd), matching tinyinfer exactly.
    int64_t Dh = hd / 2;
    rope_cos_ = Tensor({cfg.max_seq_len, Dh});
    rope_sin_ = Tensor({cfg.max_seq_len, Dh});
    for (int64_t pos = 0; pos < cfg.max_seq_len; ++pos)
        for (int64_t i = 0; i < Dh; ++i) {
            float inv_freq = 1.0f / std::pow(cfg.rope_theta,
                                             (2.0f * i) / static_cast<float>(hd));
            float angle = static_cast<float>(pos) * inv_freq;
            rope_cos_.data()[pos * Dh + i] = std::cos(angle);
            rope_sin_.data()[pos * Dh + i] = std::sin(angle);
        }
}

std::vector<Tensor*> GPTModel::parameters() {
    std::vector<Tensor*> ps;
    ps.push_back(&tok_embeddings.weight);
    for (auto& b : blocks_) {
        for (Tensor* p : b.rms_attn.parameters())
            ps.push_back(p);
        for (Tensor* p : b.wq.parameters())
            ps.push_back(p);
        for (Tensor* p : b.wk.parameters())
            ps.push_back(p);
        for (Tensor* p : b.wv.parameters())
            ps.push_back(p);
        for (Tensor* p : b.wo.parameters())
            ps.push_back(p);
        for (Tensor* p : b.rms_ffn.parameters())
            ps.push_back(p);
        for (Tensor* p : b.wgate.parameters())
            ps.push_back(p);
        for (Tensor* p : b.wup.parameters())
            ps.push_back(p);
        for (Tensor* p : b.wdown.parameters())
            ps.push_back(p);
    }
    for (Tensor* p : rms_final.parameters())
        ps.push_back(p);
    if (!tie_weights_)
        for (Tensor* p : lm_head.parameters())
            ps.push_back(p);
    return ps;
}

Tensor GPTModel::forward(const std::vector<int64_t>& idx, int64_t B, int64_t T) {
    if (static_cast<int64_t>(idx.size()) != B * T)
        throw std::invalid_argument("GPTModel::forward: idx size mismatch");
    if (T > cfg_.max_seq_len)
        throw std::invalid_argument("GPTModel::forward: sequence too long");

    int64_t n_h = cfg_.n_heads, hd = cfg_.dim / n_h;
    float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    Tensor x = tok_embeddings.forward(idx);          // [B*T, dim]
    x = reshape(x, {B, T, cfg_.dim});                // [B, T, dim]

    // causal mask [T, T]: 0 on/below diagonal, large negative above
    Tensor mask = full({T, T}, 0.0f);
    for (int64_t i = 0; i < T; ++i)
        for (int64_t j = i + 1; j < T; ++j)
            mask.data()[i * T + j] = -1e9f;

    // rope tables sliced to [T, hd/2]
    int64_t Dh = hd / 2;
    Tensor cos = full({T, Dh}, 0.0f), sin = full({T, Dh}, 0.0f);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t d = 0; d < Dh; ++d) {
            cos.data()[t * Dh + d] = rope_cos_.data()[t * Dh + d];
            sin.data()[t * Dh + d] = rope_sin_.data()[t * Dh + d];
        }

    auto drop = [&](Tensor t) {
        if (cfg_.dropout > 0.0f && training)
            return dropout(t, cfg_.dropout, ++dropout_seed_);
        return t;
    };

    for (auto& b : blocks_) {
        // attention block
        Tensor h = b.rms_attn.forward(x);                       // [B,T,dim]
        Tensor h2d = reshape(h, {B * T, cfg_.dim});
        Tensor q = reshape(b.wq.forward(h2d), {B, T, n_h, hd});
        Tensor k = reshape(b.wk.forward(h2d), {B, T, n_h, hd});
        Tensor v = reshape(b.wv.forward(h2d), {B, T, n_h, hd});
        q = rope(q, cos, sin);
        k = rope(k, cos, sin);
        // [B,T,H,D] -> [B*H,T,D] for batched matmul
        Tensor qb = reshape(permute(q, {0, 2, 1, 3}), {B * n_h, T, hd});
        Tensor kb = reshape(permute(k, {0, 2, 1, 3}), {B * n_h, T, hd});
        Tensor vb = reshape(permute(v, {0, 2, 1, 3}), {B * n_h, T, hd});
        Tensor scores = bmm(qb, transpose(kb, 1, 2));            // [B*H,T,T]
        scores = mul(scores, full(scores.shape(), scale));
        scores = add(scores, mask);                             // broadcast over batch
        Tensor probs = softmax(scores, -1);
        probs = drop(probs);
        Tensor att = bmm(probs, vb);                            // [B*H,T,hd]
        // [B*H,T,D] -> [B,H,T,D] -> [B,T,H,D] -> [B,T,H*D]: the direct
        // reshape would interleave head and time indices incorrectly.
        Tensor att_bt = reshape(permute(reshape(att, {B, n_h, T, hd}), {0, 2, 1, 3}),
                                {B, T, n_h * hd});
        Tensor att_out = b.wo.forward(reshape(att_bt, {B * T, n_h * hd}));
        x = add(x, reshape(drop(att_out), {B, T, cfg_.dim}));

        // SwiGLU feed-forward block
        Tensor hff = b.rms_ffn.forward(x);
        Tensor hff2d = reshape(hff, {B * T, cfg_.dim});
        Tensor gate = silu(b.wgate.forward(hff2d));
        Tensor up = b.wup.forward(hff2d);
        Tensor ffn = b.wdown.forward(mul(gate, up));
        x = add(x, reshape(drop(ffn), {B, T, cfg_.dim}));
    }

    Tensor hn = rms_final.forward(x);
    Tensor hn2d = reshape(hn, {B * T, cfg_.dim});
    // tied embeddings: logits = hn @ E^T
    Tensor logits = matmul(hn2d, transpose(tok_embeddings.weight, 0, 1));
    return logits; // [B*T, vocab]
}

void GPTModel::save_checkpoint(const std::string& dir) const {
    namespace fs = std::filesystem;
    fs::create_directories(dir);

    struct Named {
        std::string name;
        const Tensor* t;
    };
    std::vector<Named> tensors;
    tensors.push_back({"model.embed_tokens.weight", &tok_embeddings.weight});
    for (int64_t l = 0; l < cfg_.n_layers; ++l) {
        const Block& b = blocks_[static_cast<size_t>(l)];
        std::string pre = "model.layers." + std::to_string(l);
        tensors.push_back({pre + ".self_attn.q_proj.weight", &b.wq.weight});
        tensors.push_back({pre + ".self_attn.k_proj.weight", &b.wk.weight});
        tensors.push_back({pre + ".self_attn.v_proj.weight", &b.wv.weight});
        tensors.push_back({pre + ".self_attn.o_proj.weight", &b.wo.weight});
        tensors.push_back({pre + ".mlp.gate_proj.weight", &b.wgate.weight});
        tensors.push_back({pre + ".mlp.up_proj.weight", &b.wup.weight});
        tensors.push_back({pre + ".mlp.down_proj.weight", &b.wdown.weight});
        tensors.push_back({pre + ".input_layernorm.weight", &b.rms_attn.weight});
        tensors.push_back({pre + ".post_attention_layernorm.weight", &b.rms_ffn.weight});
    }
    tensors.push_back({"model.norm.weight", &rms_final.weight});
    if (!tie_weights_)
        tensors.push_back({"lm_head.weight", &lm_head.weight});

    // safetensors: JSON header + raw LE float32 payloads
    nlohmann::json header;
    int64_t offset = 0;
    for (auto& nt : tensors) {
        int64_t nbytes = nt.t->numel() * 4;
        nlohmann::json entry;
        entry["dtype"] = "F32";
        entry["shape"] = nt.t->shape();
        entry["data_offsets"] = {offset, offset + nbytes};
        header[nt.name] = entry;
        offset += nbytes;
    }
    std::string header_str = header.dump();

    std::ofstream f(dir + "/model.safetensors", std::ios::binary);
    if (!f)
        throw std::runtime_error("save_checkpoint: cannot open model.safetensors");
    uint64_t hlen = header_str.size();
    f.write(reinterpret_cast<const char*>(&hlen), 8);
    f.write(header_str.data(), static_cast<std::streamsize>(header_str.size()));
    for (auto& nt : tensors)
        f.write(reinterpret_cast<const char*>(nt.t->data()),
                static_cast<std::streamsize>(nt.t->numel() * 4));
    f.close();

    // tinyinfer model config
    nlohmann::json cfg;
    cfg["hidden_size"] = cfg_.dim;
    cfg["num_hidden_layers"] = cfg_.n_layers;
    cfg["num_attention_heads"] = cfg_.n_heads;
    cfg["num_key_value_heads"] = cfg_.n_kv_heads;
    cfg["vocab_size"] = cfg_.vocab_size;
    cfg["intermediate_size"] = cfg_.ffn_dim;
    cfg["max_position_embeddings"] = cfg_.max_seq_len;
    cfg["rms_norm_eps"] = cfg_.norm_eps;
    cfg["rope_theta"] = cfg_.rope_theta;
    std::ofstream cf(dir + "/config.json");
    cf << cfg.dump(2);
}

void GPTModel::load_checkpoint(const std::string& dir) {
    // Reads the safetensors written by save_checkpoint and copies weights
    // into the model's parameters. Optimizer state is not restored.
    std::string path = dir + "/model.safetensors";
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("load_checkpoint: cannot open " + path);
    uint64_t hlen = 0;
    f.read(reinterpret_cast<char*>(&hlen), 8);
    std::string hstr(static_cast<size_t>(hlen), '\0');
    f.read(hstr.data(), static_cast<std::streamsize>(hlen));
    nlohmann::json header = nlohmann::json::parse(hstr);

    struct Named {
        std::string name;
        Tensor* t;
    };
    std::vector<Named> tensors;
    tensors.push_back({"model.embed_tokens.weight", &tok_embeddings.weight});
    for (int64_t l = 0; l < cfg_.n_layers; ++l) {
        Block& b = blocks_[static_cast<size_t>(l)];
        std::string pre = "model.layers." + std::to_string(l);
        tensors.push_back({pre + ".self_attn.q_proj.weight", &b.wq.weight});
        tensors.push_back({pre + ".self_attn.k_proj.weight", &b.wk.weight});
        tensors.push_back({pre + ".self_attn.v_proj.weight", &b.wv.weight});
        tensors.push_back({pre + ".self_attn.o_proj.weight", &b.wo.weight});
        tensors.push_back({pre + ".mlp.gate_proj.weight", &b.wgate.weight});
        tensors.push_back({pre + ".mlp.up_proj.weight", &b.wup.weight});
        tensors.push_back({pre + ".mlp.down_proj.weight", &b.wdown.weight});
        tensors.push_back({pre + ".input_layernorm.weight", &b.rms_attn.weight});
        tensors.push_back({pre + ".post_attention_layernorm.weight", &b.rms_ffn.weight});
    }
    tensors.push_back({"model.norm.weight", &rms_final.weight});
    if (!tie_weights_)
        tensors.push_back({"lm_head.weight", &lm_head.weight});

    for (auto& nt : tensors) {
        auto it = header.find(nt.name);
        if (it == header.end())
            throw std::runtime_error("load_checkpoint: missing tensor " + nt.name);
        auto data_off = (*it)["data_offsets"];
        uint64_t beg = data_off[0].get<uint64_t>();
        uint64_t end = data_off[1].get<uint64_t>();
        if (end - beg != static_cast<uint64_t>(nt.t->numel() * 4))
            throw std::runtime_error("load_checkpoint: size mismatch for " + nt.name);
        f.seekg(static_cast<std::streamoff>(8 + hlen + beg), std::ios::beg);
        f.read(reinterpret_cast<char*>(nt.t->data()),
               static_cast<std::streamsize>(nt.t->numel() * 4));
        if (!f)
            throw std::runtime_error("load_checkpoint: read failed for " + nt.name);
    }
}

} // namespace tt
