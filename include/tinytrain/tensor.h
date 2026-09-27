#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace tt {

// Row-major dense float32 tensor.
//
// This is the storage + shape foundation. Autograd (the computation graph,
// reverse-mode differentiation) is built on top of this in the next step;
// nothing here knows about gradients yet.
class Tensor {
public:
    Tensor() = default;
    explicit Tensor(std::vector<int64_t> shape);
    Tensor(std::vector<int64_t> shape, float fill);

    const std::vector<int64_t>& shape() const { return shape_; }
    int64_t ndim() const { return static_cast<int64_t>(shape_.size()); }
    int64_t numel() const { return numel_; }
    int64_t dim(int64_t i) const { return shape_.at(static_cast<size_t>(i)); }

    float* data() { return data_->data(); }
    const float* data() const { return data_->data(); }

    // Row-major offset of a multi-index.
    int64_t offset(const std::vector<int64_t>& idx) const;

private:
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    int64_t numel_ = 0;
    std::shared_ptr<std::vector<float>> data_;
};

} // namespace tt
