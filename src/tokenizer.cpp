// BPE tokenizer implementation.
//
// Matches the Hugging Face `tokenizers` BPE model behavior for Llama-style
// tokenizers: U+2581 normalizer, char/byte-fallback initial split, merges
// applied greedily by rank, added special tokens matched before BPE.

#include "tinytrain/tokenizer.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace tt {
namespace {

// U+2581 (▁) in UTF-8, the SentencePiece space marker.
const char* kSpSpace = "\xE2\x96\x81";

// Length of the UTF-8 sequence starting at p (no validation; well-formed input).
size_t utf8_len(unsigned char c) {
    if ((c & 0x80) == 0) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    return 4;
}

bool is_byte_fallback_token(const std::string& tok, unsigned* byte_out) {
    // "<0xNN>" with uppercase hex, exactly 6 chars.
    if (tok.size() != 6 || tok[0] != '<' || tok[1] != '0' || tok[2] != 'x' || tok[5] != '>')
        return false;
    unsigned v = 0;
    for (int i = 3; i <= 4; ++i) {
        char c = tok[i];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= c - '0';
        else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
        else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
        else return false;
    }
    *byte_out = v;
    return true;
}

} // namespace

void Tokenizer::load(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("tokenizer: cannot open " + path);
    nlohmann::json j;
    f >> j;

    const auto& model = j.at("model");
    if (model.at("type").get<std::string>() != "BPE")
        throw std::runtime_error("tokenizer: only BPE supported, got " +
                                 model.at("type").get<std::string>());

    const auto& vocab = model.at("vocab");
    id_to_token_.assign(vocab.size(), "");
    for (auto it = vocab.begin(); it != vocab.end(); ++it) {
        int id = it.value().get<int>();
        if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size())
            throw std::runtime_error("tokenizer: vocab id out of range");
        id_to_token_[id] = it.key();
        token_to_id_[it.key()] = id;
    }
    for (size_t i = 0; i < id_to_token_.size(); ++i)
        if (id_to_token_[i].empty())
            throw std::runtime_error("tokenizer: missing vocab entry for id " +
                                     std::to_string(i));

    int rank = 0;
    for (const auto& m : model.at("merges")) {
        std::string s = m.get<std::string>();
        size_t sp = s.find(' ');
        if (sp == std::string::npos || sp == 0 || sp + 1 >= s.size())
            throw std::runtime_error("tokenizer: bad merge entry");
        std::string key = s.substr(0, sp) + '\x1f' + s.substr(sp + 1);
        if (!merge_rank_.count(key)) merge_rank_[key] = rank;
        ++rank;
    }

    byte_fallback_ = model.value("byte_fallback", false);
    fuse_unk_ = model.value("fuse_unk", false);

    specials_.clear();
    for (const auto& a : j.value("added_tokens", nlohmann::json::array())) {
        AddedToken t;
        t.raw = a.at("content").get<std::string>();
        t.norm = normalize(t.raw);
        t.id = a.at("id").get<int>();
        t.normalized = a.value("normalized", false);
        specials_.push_back(t);
        if (t.raw == "<s>") bos_id_ = t.id;
        if (t.raw == "</s>") eos_id_ = t.id;
        if (t.raw == "<unk>") unk_id_ = t.id;
    }
    std::sort(specials_.begin(), specials_.end(), [](const AddedToken& x, const AddedToken& y) {
        size_t xn = x.normalized ? x.norm.size() : x.raw.size();
        size_t yn = y.normalized ? y.norm.size() : y.raw.size();
        return xn > yn;
    });
}

// Splits `text` on special tokens (longest match wins), BPE-encoding the gaps.
// `match` selects which special-token spellings to look for: normalized
// (already-normalized input, pieces encoded raw) or raw (pieces normalized
// per piece). `bpe_raw` tells bpe_encode_piece whether pieces skip normalization.
void Tokenizer::split_encode(const std::string& text,
                             const std::vector<const AddedToken*>& match,
                             bool bpe_raw, std::vector<int>& ids) const {
    size_t pos = 0;
    while (pos < text.size()) {
        bool matched = false;
        for (const AddedToken* t : match) {
            const std::string& m = t->normalized ? t->norm : t->raw;
            if (text.compare(pos, m.size(), m) == 0) {
                ids.push_back(t->id);
                pos += m.size();
                matched = true;
                break;
            }
        }
        if (matched) continue;
        size_t next = text.size();
        for (const AddedToken* t : match) {
            const std::string& m = t->normalized ? t->norm : t->raw;
            size_t p = text.find(m, pos);
            if (p != std::string::npos && p < next) next = p;
        }
        auto piece = bpe_encode_piece(text.substr(pos, next - pos), bpe_raw);
        ids.insert(ids.end(), piece.begin(), piece.end());
        pos = next;
    }
}

std::vector<int> Tokenizer::encode(const std::string& text, bool add_bos) const {
    if (!is_loaded()) throw std::runtime_error("tokenizer: not loaded");
    std::vector<int> ids;
    if (add_bos) ids.push_back(bos_id_);

    // Partition specials by their `normalized` flag: normalized=true tokens
    // are matched in normalized text (HF behavior for Llama tokenizers);
    // normalized=false tokens are matched in the raw text. Each group is
    // already sorted longest-first by its match spelling.
    std::vector<const AddedToken*> norm_group, raw_group;
    for (const auto& t : specials_) {
        if (t.normalized)
            norm_group.push_back(&t);
        else
            raw_group.push_back(&t);
    }

    if (!norm_group.empty() && !raw_group.empty()) {
        // Mixed (unusual): handle raw group first on the raw text, then the
        // normalized group on the normalized remainder. Not exercised by our
        // tokenizers; kept simple and documented.
        throw std::runtime_error("tokenizer: mixed normalized/raw added tokens unsupported");
    }
    if (text.empty()) return ids; // HF short-circuits: empty in -> empty out
    if (!norm_group.empty()) {
        split_encode(normalize(text), norm_group, /*bpe_raw=*/true, ids);
    } else {
        split_encode(text, raw_group, /*bpe_raw=*/false, ids);
    }
    return ids;
}

std::string Tokenizer::normalize(const std::string& text) {
    std::string norm;
    norm.reserve(text.size() + 3);
    norm += kSpSpace;
    for (char c : text) {
        if (c == ' ')
            norm += kSpSpace;
        else
            norm += c;
    }
    return norm;
}

std::vector<int> Tokenizer::bpe_encode_piece(const std::string& piece,
                                             bool already_normalized) const {
    // 1. Normalizer: prepend ▁, replace ' ' with ▁ (unless already done).
    //    (For normalized=false added tokens, splitting happens on the raw text
    //    first, so each piece is normalized independently — verified vs HF.)
    const std::string norm = already_normalized ? piece : normalize(piece);

    // 2. Initial split: chars, or UTF-8 bytes as <0xNN> when the char is OOV.
    std::vector<std::string> toks;
    for (size_t i = 0; i < norm.size();) {
        size_t n = utf8_len(static_cast<unsigned char>(norm[i]));
        std::string ch = norm.substr(i, n);
        i += n;
        if (token_to_id_.count(ch)) {
            toks.push_back(ch);
        } else if (byte_fallback_) {
            for (size_t k = 0; k < n; ++k) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "<0x%02X>", (unsigned char)ch[k]);
                toks.push_back(buf);
            }
        } else {
            toks.push_back(id_to_token_[unk_id_]);
        }
    }

    // 3. Greedy merges by rank (lower rank merges first).
    const int INF = 1 << 30;
    while (toks.size() > 1) {
        int best = -1, best_rank = INF;
        for (size_t i = 0; i + 1 < toks.size(); ++i) {
            auto it = merge_rank_.find(toks[i] + '\x1f' + toks[i + 1]);
            int r = (it == merge_rank_.end()) ? INF : it->second;
            if (r < best_rank) {
                best_rank = r;
                best = static_cast<int>(i);
            }
        }
        if (best < 0) break;
        toks[best] += toks[best + 1];
        toks.erase(toks.begin() + best + 1);
    }

    // 4. Map pieces to ids.
    std::vector<int> ids;
    ids.reserve(toks.size());
    for (const auto& t : toks) {
        auto it = token_to_id_.find(t);
        ids.push_back(it == token_to_id_.end() ? unk_id_ : it->second);
    }
    if (fuse_unk_) {
        std::vector<int> fused;
        for (int id : ids) {
            if (id == unk_id_ && !fused.empty() && fused.back() == unk_id_) continue;
            fused.push_back(id);
        }
        ids.swap(fused);
    }
    return ids;
}

std::string Tokenizer::decode_token(int id) const {
    if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size())
        throw std::out_of_range("tokenizer: id out of range");
    unsigned byte = 0;
    if (is_byte_fallback_token(id_to_token_[id], &byte))
        return std::string(1, static_cast<char>(byte));
    return id_to_token_[id];
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids) {
        // Skip special tokens (bos/eos/unk and any other added tokens).
        bool special = false;
        for (const auto& t : specials_)
            if (t.id == id) {
                special = true;
                break;
            }
        if (special) continue;
        out += decode_token(id);
    }
    // ▁ -> ' ', then strip exactly one leading space (from the prepended ▁).
    std::string res;
    res.reserve(out.size());
    for (size_t i = 0; i < out.size();) {
        if (out.compare(i, 3, kSpSpace) == 0) {
            res += ' ';
            i += 3;
        } else {
            res += out[i++];
        }
    }
    if (!res.empty() && res[0] == ' ') res.erase(res.begin());
    return res;
}

} // namespace tt
