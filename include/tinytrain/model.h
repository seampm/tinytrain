#pragma once

// A Llama-architecture decoder-only transformer built from the nn blocks.
// The checkpoint it exports loads directly into tinyinfer: same tensor names,
// same [out, in] weight layout, same RoPE convention.

#include <cstdint>
#include <string>
#include <vector>

#include "tinytrain/nn.h"

namespace tt {

struct GPTConfig {
    int64_t vocab_size = 32000;
    int64_t dim = 256;
    int64_t n_layers = 6;
    int64_t n_heads = 8;
    int64_t n_kv_heads = 8; // kept equal to n_heads (no GQA in the demo model)
    int64_t ffn_dim = 768;
    int64_t max_seq_len = 256;
    float norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
    float dropout = 0.0f;
};

class GPTModel : public Module {
public:
    explicit GPTModel(const GPTConfig& cfg, uint64_t seed);

    // idx: B*T token ids (row-major [B, T]); returns logits [B*T, vocab_size].
    Tensor forward(const std::vector<int64_t>& idx, int64_t B, int64_t T);

    std::vector<Tensor*> parameters() override;

    // Writes <dir>/model.safetensors (F32, tinyinfer tensor names) and
    // <dir>/config.json (tinyinfer model config).
    void save_checkpoint(const std::string& dir) const;
    void load_checkpoint(const std::string& dir);

    const GPTConfig& config() const { return cfg_; }

private:
    struct Block {
        RMSNorm rms_attn{1, 1e-5f};
        Linear wq{1, 1, 0}, wk{1, 1, 0}, wv{1, 1, 0}, wo{1, 1, 0};
        RMSNorm rms_ffn{1, 1e-5f};
        Linear wgate{1, 1, 0}, wup{1, 1, 0}, wdown{1, 1, 0};
    };

    GPTConfig cfg_;
    Embedding tok_embeddings{1, 1, 0};
    std::vector<Block> blocks_;
    RMSNorm rms_final{1, 1e-5f};
    Linear lm_head{1, 1, 0}; // [vocab, dim]; tied to embeddings when tie_weights
    bool tie_weights_ = true;

    Tensor rope_cos_; // [max_seq_len, head_dim/2], constants
    Tensor rope_sin_;
    uint64_t dropout_seed_ = 0;
};

} // namespace tt
