// Tokenize a text corpus into a flat binary file of int32 token ids.
//
// Usage: prepare_data <tokenizer.json> <input.txt> <output.tokens>
//
// Documents are split on blank lines (TinyStories format: one story per
// block); each document is encoded and followed by the EOS token so the
// model learns document boundaries.
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "tinytrain/tokenizer.h"

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: prepare_data <tokenizer.json> <input.txt> <output.tokens>\n";
        return 1;
    }
    tt::Tokenizer tok;
    tok.load(argv[1]);
    std::cout << "vocab: " << tok.vocab_size() << "\n";

    std::ifstream in(argv[2]);
    if (!in) {
        std::cerr << "cannot open " << argv[2] << "\n";
        return 1;
    }
    std::ofstream out(argv[3], std::ios::binary);
    if (!out) {
        std::cerr << "cannot open " << argv[3] << "\n";
        return 1;
    }

    int64_t total = 0, docs = 0;
    std::string line, doc;
    auto flush_doc = [&] {
        if (doc.empty())
            return;
        std::vector<int> ids = tok.encode(doc, false);
        for (int id : ids) {
            int32_t v = id;
            out.write(reinterpret_cast<const char*>(&v), 4);
        }
        int32_t eos = tok.eos_id();
        out.write(reinterpret_cast<const char*>(&eos), 4);
        total += static_cast<int64_t>(ids.size()) + 1;
        ++docs;
        doc.clear();
    };
    while (std::getline(in, line)) {
        // blank line (or <|endoftext|>) ends a document
        if (line.empty() || line == "<|endoftext|>") {
            flush_doc();
        } else {
            if (!doc.empty())
                doc += "\n";
            doc += line;
        }
    }
    flush_doc();
    std::cout << "documents: " << docs << ", tokens: " << total << "\n";
    return 0;
}
