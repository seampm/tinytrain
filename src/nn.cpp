#include "tinytrain/nn.h"

#include <cmath>

#include "tinytrain/ops.h"

namespace tt {

void Module::zero_grad() {
    for (Tensor* p : parameters())
        p->zero_grad();
}

Linear::Linear(int64_t in_features, int64_t out_features, uint64_t seed)
    : weight({out_features, in_features}, true) {
    // Xavier-style: Var(W) = 2 / (in + out)
    float scale = std::sqrt(2.0f / static_cast<float>(in_features + out_features));
    Tensor init = randn({out_features, in_features}, seed);
    for (int64_t i = 0; i < init.numel(); ++i)
        weight.data()[i] = init.data()[i] * scale;
}

Tensor Linear::forward(const Tensor& x) {
    // x [N, in] @ W^T [in, out]
    return matmul(x, transpose(weight, 0, 1));
}

Embedding::Embedding(int64_t vocab_size, int64_t dim, uint64_t seed)
    : weight({vocab_size, dim}, true) {
    Tensor init = randn({vocab_size, dim}, seed);
    float scale = 1.0f / std::sqrt(static_cast<float>(dim));
    for (int64_t i = 0; i < init.numel(); ++i)
        weight.data()[i] = init.data()[i] * scale;
}

Tensor Embedding::forward(const std::vector<int64_t>& idx) {
    return embedding(weight, idx);
}

RMSNorm::RMSNorm(int64_t dim, float eps) : weight({dim}, 1.0f, true), eps(eps) {}

Tensor RMSNorm::forward(const Tensor& x) {
    return rmsnorm(x, weight, eps);
}

} // namespace tt
