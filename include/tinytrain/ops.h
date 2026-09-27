#pragma once

// Elementwise, shape, linear-algebra, and neural-network ops with reverse-mode
// rules. Composite ops (softmax, cross-entropy, rmsnorm) are built from
// primitives wherever the math is clean, so their gradients come for free.

#include <cstdint>
#include <vector>

#include "tinytrain/tensor.h"

namespace tt {

// ---- elementwise (NumPy-style trailing broadcasting) ----
Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);
Tensor neg(const Tensor& a);

// ---- unary ----
Tensor exp(const Tensor& a);
Tensor log(const Tensor& a);
Tensor sqrt(const Tensor& a);
Tensor tanh(const Tensor& a);
Tensor silu(const Tensor& a); // x * sigmoid(x)
Tensor sin(const Tensor& a);
Tensor cos(const Tensor& a);
Tensor pow(const Tensor& a, float p);

// ---- reductions ----
Tensor sum(const Tensor& a, int64_t dim, bool keepdim = false);
Tensor sum_all(const Tensor& a);
Tensor mean(const Tensor& a);

// ---- shape ----
Tensor reshape(const Tensor& a, const std::vector<int64_t>& shape);
Tensor transpose(const Tensor& a, int64_t d0, int64_t d1);
Tensor permute(const Tensor& a, const std::vector<int64_t>& dims);
Tensor expand(const Tensor& a, const std::vector<int64_t>& shape);
Tensor squeeze(const Tensor& a, int64_t dim);
Tensor unsqueeze(const Tensor& a, int64_t dim);

// ---- linear algebra ----
Tensor matmul(const Tensor& a, const Tensor& b); // 2D @ 2D
Tensor bmm(const Tensor& a, const Tensor& b);    // [B,M,K] @ [B,K,N]

// ---- neural nets ----
Tensor embedding(const Tensor& weight, const std::vector<int64_t>& idx); // [V,D] -> [N,D]
Tensor softmax(const Tensor& a, int64_t dim);
Tensor log_softmax(const Tensor& a, int64_t dim);
// mean negative log-likelihood; log_probs [N,C]
Tensor nll_loss(const Tensor& log_probs, const std::vector<int64_t>& targets);
Tensor cross_entropy(const Tensor& logits, const std::vector<int64_t>& targets);
// x normalized over last dim, scaled by weight; eps for stability
Tensor rmsnorm(const Tensor& a, const Tensor& weight, float eps);
Tensor dropout(const Tensor& a, float p, uint64_t seed);
// Rotary position embeddings: x [B,T,H,D] (D even), cos/sin [T,D/2]
Tensor rope(const Tensor& x, const Tensor& cos, const Tensor& sin);

} // namespace tt
