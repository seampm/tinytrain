// Adapted from the tinyinfer tokenizer (same author): BPE from tokenizer.json.
#pragma once

// BPE tokenizer parsed from Hugging Face tokenizer.json.
//
// Supports the SentencePiece-style BPE used by Llama-family tokenizers:
//   normalizer: prepend U+2581 (▁), replace ' ' with ▁
//   encode: split out added special tokens, then byte/char-level BPE by merge rank
//   decode: ▁ -> ' ', <0xNN> byte fallback, fuse, strip one leading space

#include <string>
#include <unordered_map>
#include <vector>

namespace tt {

class Tokenizer {
public:
    void load(const std::string& tokenizer_json_path); // throws on failure
    bool is_loaded() const { return !id_to_token_.empty(); }

    // BPE-encode text; prepends bos_id() when add_bos is true.
    std::vector<int> encode(const std::string& text, bool add_bos) const;
    // Full decode: skips special tokens, ▁->space, byte fallback, strips 1 leading space.
    std::string decode(const std::vector<int>& ids) const;
    // Raw bytes for one token (no ▁ conversion, no stripping). For streaming print.
    std::string decode_token(int id) const;

    int bos_id() const { return bos_id_; }
    int eos_id() const { return eos_id_; }
    int unk_id() const { return unk_id_; }
    int vocab_size() const { return static_cast<int>(id_to_token_.size()); }

private:
    struct AddedToken {
        std::string raw;   // as listed in tokenizer.json
        std::string norm;  // normalizer applied (for normalized=true matching)
        int id;
        bool normalized;
    };
    // BPE-encode one piece. When already_normalized=false, the ▁ normalizer
    // (prepend ▁, ' ' -> ▁) runs first; when true the piece is encoded raw.
    std::vector<int> bpe_encode_piece(const std::string& piece, bool already_normalized) const;
    static std::string normalize(const std::string& text);
    // Split text on special tokens (longest match wins), BPE-encoding gaps.
    void split_encode(const std::string& text,
                      const std::vector<const AddedToken*>& match,
                      bool bpe_raw, std::vector<int>& ids) const;

    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<std::string, int> merge_rank_; // "a\x1fb" -> rank (lower = first)
    std::vector<AddedToken> specials_; // longest-first by match string
    bool byte_fallback_ = true;
    bool fuse_unk_ = true;
    int bos_id_ = 1;
    int eos_id_ = 2;
    int unk_id_ = 0;
};

} // namespace tt
