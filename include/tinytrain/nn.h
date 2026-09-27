#pragma once

// Neural network building blocks on top of the autograd engine.

#include <cstdint>
#include <vector>

#include "tinytrain/tensor.h"

namespace tt {

struct Module {
    virtual ~Module() = default;
    virtual std::vector<Tensor*> parameters() = 0;
    void zero_grad();
    bool training = true; // false => dropout becomes identity
};

// y = x @ W^T ; W: [out, in] (matches the tinyinfer checkpoint layout)
class Linear : public Module {
public:
    Linear(int64_t in_features, int64_t out_features, uint64_t seed);
    Tensor forward(const Tensor& x); // [N, in] -> [N, out]
    std::vector<Tensor*> parameters() override { return {&weight}; }
    Tensor weight;
};

// Token embedding table.
class Embedding : public Module {
public:
    Embedding(int64_t vocab_size, int64_t dim, uint64_t seed);
    Tensor forward(const std::vector<int64_t>& idx); // [N] ids -> [N, dim]
    std::vector<Tensor*> parameters() override { return {&weight}; }
    Tensor weight;
};

// RMSNorm over the last dim, scaled by a learned weight.
class RMSNorm : public Module {
public:
    RMSNorm(int64_t dim, float eps);
    Tensor forward(const Tensor& x);
    std::vector<Tensor*> parameters() override { return {&weight}; }
    Tensor weight;
    float eps;
};

} // namespace tt
