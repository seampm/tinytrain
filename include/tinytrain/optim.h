#pragma once

// Gradient-based optimizers. Operate in-place on parameter storage;
// they never touch the computation graph.

#include <cstdint>
#include <vector>

#include "tinytrain/tensor.h"

namespace tt {

// AdamW: Adam with decoupled weight decay (the modern default).
class AdamW {
public:
    AdamW(const std::vector<Tensor*>& params, float lr, float beta1 = 0.9f,
          float beta2 = 0.999f, float eps = 1e-8f, float weight_decay = 0.0f);
    void step();
    void zero_grad();
    void set_lr(float lr) { lr_ = lr; }

private:
    std::vector<Tensor*> params_;
    float lr_, beta1_, beta2_, eps_, weight_decay_;
    int64_t step_ = 0;
    // per-parameter state, allocated lazily, keyed by index into params_
    std::vector<std::vector<float>> m_, v_;
};

// SGD with optional momentum and weight decay.
class SGD {
public:
    SGD(const std::vector<Tensor*>& params, float lr, float momentum = 0.0f,
        float weight_decay = 0.0f);
    void step();
    void zero_grad();

private:
    std::vector<Tensor*> params_;
    float lr_, momentum_, weight_decay_;
    std::vector<std::vector<float>> vel_;
};

} // namespace tt
