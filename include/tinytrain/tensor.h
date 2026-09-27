#pragma once

// tinytrain autograd: reverse-mode automatic differentiation over a dynamic
// computation graph, in the spirit of PyTorch's eager execution.
//
// A Tensor is a cheap handle (shared_ptr) to an Impl holding storage, shape,
// an optional accumulated gradient, and the graph edges (prev nodes + a
// backward closure) for the op that produced it. Graphs are rebuilt on every
// forward pass; backward() topologically sorts from the loss and runs each
// node's closure in reverse order, accumulating gradients.

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace tt {

class Tensor {
public:
    struct Impl {
        std::vector<int64_t> shape;
        std::vector<int64_t> strides; // row-major
        int64_t numel = 0;
        std::vector<float> data;

        bool requires_grad = false;
        std::vector<float> grad; // accumulated gradient, empty until first

        std::function<void()> backward_fn = [] {};
        std::vector<std::shared_ptr<Impl>> prev;
    };

    Tensor() = default;
    explicit Tensor(std::vector<int64_t> shape, bool requires_grad = false);
    Tensor(std::vector<int64_t> shape, float fill, bool requires_grad = false);

    static Tensor from_impl(std::shared_ptr<Impl> impl);

    // ---- storage / shape ----
    const std::vector<int64_t>& shape() const { return impl_->shape; }
    int64_t ndim() const { return static_cast<int64_t>(impl_->shape.size()); }
    int64_t numel() const { return impl_->numel; }
    int64_t dim(int64_t i) const { return impl_->shape.at(static_cast<size_t>(i)); }
    float* data() { return impl_->data.data(); }
    const float* data() const { return impl_->data.data(); }
    int64_t offset(const std::vector<int64_t>& idx) const;

    // ---- autograd ----
    bool requires_grad() const { return impl_->requires_grad; }
    void set_requires_grad(bool v);
    bool has_grad() const { return !impl_->grad.empty(); }
    float* grad_data() { return impl_->grad.data(); }
    const float* grad_data() const { return impl_->grad.data(); }
    void zero_grad();

    // Reverse-mode differentiation. Must be called on a scalar tensor.
    // Accumulates gradients into every requires_grad leaf/input.
    void backward();

    // Same storage, detached from the graph.
    Tensor detach() const;

    std::shared_ptr<Impl> impl() { return impl_; }
    const std::shared_ptr<Impl>& impl() const { return impl_; }

private:
    std::shared_ptr<Impl> impl_;
};

// ---- construction helpers ----
Tensor zeros(const std::vector<int64_t>& shape, bool requires_grad = false);
Tensor ones(const std::vector<int64_t>& shape, bool requires_grad = false);
Tensor full(const std::vector<int64_t>& shape, float v, bool requires_grad = false);
// Standard-normal samples, deterministic for a given seed.
Tensor randn(const std::vector<int64_t>& shape, uint64_t seed);
// Uniform samples in [lo, hi), deterministic for a given seed.
Tensor rand_uniform(const std::vector<int64_t>& shape, float lo, float hi, uint64_t seed);

} // namespace tt
