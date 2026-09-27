#include "tinytrain/tensor.h"

#include <cmath>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace tt {

namespace {

std::shared_ptr<Tensor::Impl> make_impl(std::vector<int64_t> shape, bool requires_grad) {
    if (shape.empty())
        throw std::invalid_argument("Tensor: shape must have at least one dim");
    auto impl = std::make_shared<Tensor::Impl>();
    impl->shape = std::move(shape);
    impl->strides.resize(impl->shape.size());
    int64_t stride = 1;
    for (int64_t i = static_cast<int64_t>(impl->shape.size()) - 1; i >= 0; --i) {
        impl->strides[static_cast<size_t>(i)] = stride;
        stride *= impl->shape[static_cast<size_t>(i)];
    }
    impl->numel = stride;
    for (int64_t d : impl->shape)
        if (d <= 0)
            throw std::invalid_argument("Tensor: dims must be positive");
    impl->data.assign(static_cast<size_t>(impl->numel), 0.0f);
    impl->requires_grad = requires_grad;
    return impl;
}

} // namespace

Tensor::Tensor(std::vector<int64_t> shape, bool requires_grad)
    : impl_(make_impl(std::move(shape), requires_grad)) {}

Tensor::Tensor(std::vector<int64_t> shape, float fill, bool requires_grad)
    : Tensor(std::move(shape), requires_grad) {
    std::fill(impl_->data.begin(), impl_->data.end(), fill);
}

Tensor Tensor::from_impl(std::shared_ptr<Impl> impl) {
    Tensor t;
    t.impl_ = std::move(impl);
    return t;
}

int64_t Tensor::offset(const std::vector<int64_t>& idx) const {
    if (idx.size() != impl_->shape.size())
        throw std::invalid_argument("Tensor::offset: rank mismatch");
    int64_t off = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (idx[i] < 0 || idx[i] >= impl_->shape[i])
            throw std::out_of_range("Tensor::offset: index out of range");
        off += idx[i] * impl_->strides[i];
    }
    return off;
}

void Tensor::set_requires_grad(bool v) {
    impl_->requires_grad = v;
    if (!v)
        impl_->grad.clear();
}

void Tensor::zero_grad() {
    if (!impl_->requires_grad)
        return;
    impl_->grad.assign(static_cast<size_t>(impl_->numel), 0.0f);
}

Tensor Tensor::detach() const {
    Tensor t;
    t.impl_ = std::make_shared<Impl>(*impl_);
    t.impl_->requires_grad = false;
    t.impl_->grad.clear();
    t.impl_->backward_fn = [] {};
    t.impl_->prev.clear();
    return t;
}

void Tensor::backward() {
    if (impl_->numel != 1)
        throw std::invalid_argument("Tensor::backward: only scalar tensors");

    // Topological order via DFS over prev edges.
    std::vector<std::shared_ptr<Impl>> topo;
    std::unordered_set<Impl*> visited;
    std::function<void(const std::shared_ptr<Impl>&)> dfs =
        [&](const std::shared_ptr<Impl>& n) {
            if (!visited.insert(n.get()).second)
                return;
            for (auto& p : n->prev)
                dfs(p);
            topo.push_back(n);
        };
    dfs(impl_);

    // Fresh backward pass: intermediate grads from a previous pass are stale,
    // so clear them. Leaf grads (nodes with no prev) intentionally accumulate
    // across backward() calls, matching the standard convention.
    for (auto& n : topo) {
        if (!n->prev.empty())
            n->grad.clear();
    }

    // Seed: d(loss)/d(loss) = 1.
    impl_->grad.assign(1, 1.0f);

    for (auto it = topo.rbegin(); it != topo.rend(); ++it)
        (*it)->backward_fn();
}

// ---- construction helpers ----

Tensor zeros(const std::vector<int64_t>& shape, bool requires_grad) {
    return Tensor(shape, 0.0f, requires_grad);
}

Tensor ones(const std::vector<int64_t>& shape, bool requires_grad) {
    return Tensor(shape, 1.0f, requires_grad);
}

Tensor full(const std::vector<int64_t>& shape, float v, bool requires_grad) {
    return Tensor(shape, v, requires_grad);
}

namespace {
// Deterministic xorshift64* RNG (no <random> distribution overhead, stable
// across platforms for a given seed).
struct Rng {
    explicit Rng(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    uint64_t next() {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return s * 0x2545F4914F6CDD1DULL;
    }
    // [0, 1)
    double uniform01() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
    // standard normal via Box-Muller
    double normal() {
        double u1 = uniform01(), u2 = uniform01();
        return std::sqrt(-2.0 * std::log(u1 + 1e-300)) *
               std::cos(2.0 * 3.141592653589793 * u2);
    }
    uint64_t s;
};
} // namespace

Tensor randn(const std::vector<int64_t>& shape, uint64_t seed) {
    Tensor t(shape);
    Rng rng(seed);
    for (int64_t i = 0; i < t.numel(); ++i)
        t.data()[i] = static_cast<float>(rng.normal());
    return t;
}

Tensor rand_uniform(const std::vector<int64_t>& shape, float lo, float hi, uint64_t seed) {
    Tensor t(shape);
    Rng rng(seed);
    for (int64_t i = 0; i < t.numel(); ++i)
        t.data()[i] = lo + static_cast<float>(rng.uniform01()) * (hi - lo);
    return t;
}

} // namespace tt
