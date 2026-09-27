#include "tinytrain/tensor.h"

#include <numeric>
#include <stdexcept>

namespace tt {

Tensor::Tensor(std::vector<int64_t> shape) : shape_(std::move(shape)) {
    if (shape_.empty())
        throw std::invalid_argument("Tensor: shape must have at least one dim");
    for (int64_t d : shape_)
        if (d <= 0)
            throw std::invalid_argument("Tensor: dims must be positive");
    strides_.resize(shape_.size());
    int64_t stride = 1;
    for (int64_t i = static_cast<int64_t>(shape_.size()) - 1; i >= 0; --i) {
        strides_[static_cast<size_t>(i)] = stride;
        stride *= shape_[static_cast<size_t>(i)];
    }
    numel_ = stride;
    data_ = std::make_shared<std::vector<float>>(static_cast<size_t>(numel_), 0.0f);
}

Tensor::Tensor(std::vector<int64_t> shape, float fill) : Tensor(std::move(shape)) {
    std::fill(data_->begin(), data_->end(), fill);
}

int64_t Tensor::offset(const std::vector<int64_t>& idx) const {
    if (idx.size() != shape_.size())
        throw std::invalid_argument("Tensor::offset: rank mismatch");
    int64_t off = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (idx[i] < 0 || idx[i] >= shape_[i])
            throw std::out_of_range("Tensor::offset: index out of range");
        off += idx[i] * strides_[i];
    }
    return off;
}

} // namespace tt
