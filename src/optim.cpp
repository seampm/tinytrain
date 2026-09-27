#include "tinytrain/optim.h"

#include <cmath>

namespace tt {

AdamW::AdamW(const std::vector<Tensor*>& params, float lr, float beta1, float beta2,
             float eps, float weight_decay)
    : params_(params), lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps),
      weight_decay_(weight_decay) {}

void AdamW::zero_grad() {
    for (Tensor* p : params_)
        p->zero_grad();
}

void AdamW::step() {
    ++step_;
    if (m_.empty()) {
        m_.resize(params_.size());
        v_.resize(params_.size());
    }
    float bc1 = 1.0f - std::pow(beta1_, static_cast<float>(step_));
    float bc2 = 1.0f - std::pow(beta2_, static_cast<float>(step_));
    float lr_t = lr_ * std::sqrt(bc2) / bc1; // bias correction folded into lr
    for (size_t pi = 0; pi < params_.size(); ++pi) {
        Tensor* p = params_[pi];
        if (!p->has_grad())
            continue;
        int64_t n = p->numel();
        if (m_[pi].empty()) {
            m_[pi].assign(static_cast<size_t>(n), 0.0f);
            v_[pi].assign(static_cast<size_t>(n), 0.0f);
        }
        float* d = p->data();
        const float* g = p->grad_data();
        float* m = m_[pi].data();
        float* v = v_[pi].data();
        for (int64_t i = 0; i < n; ++i) {
            // decoupled weight decay
            d[i] -= lr_ * weight_decay_ * d[i];
            float gi = g[i];
            m[i] = beta1_ * m[i] + (1.0f - beta1_) * gi;
            v[i] = beta2_ * v[i] + (1.0f - beta2_) * gi * gi;
            d[i] -= lr_t * m[i] / (std::sqrt(v[i]) + eps_);
        }
    }
}

SGD::SGD(const std::vector<Tensor*>& params, float lr, float momentum, float weight_decay)
    : params_(params), lr_(lr), momentum_(momentum), weight_decay_(weight_decay) {}

void SGD::zero_grad() {
    for (Tensor* p : params_)
        p->zero_grad();
}

void SGD::step() {
    if (momentum_ != 0.0f && vel_.empty())
        vel_.resize(params_.size());
    for (size_t pi = 0; pi < params_.size(); ++pi) {
        Tensor* p = params_[pi];
        if (!p->has_grad())
            continue;
        int64_t n = p->numel();
        float* d = p->data();
        const float* g = p->grad_data();
        if (momentum_ == 0.0f) {
            for (int64_t i = 0; i < n; ++i)
                d[i] -= lr_ * (g[i] + weight_decay_ * d[i]);
        } else {
            if (vel_[pi].empty())
                vel_[pi].assign(static_cast<size_t>(n), 0.0f);
            float* vel = vel_[pi].data();
            for (int64_t i = 0; i < n; ++i) {
                float gi = g[i] + weight_decay_ * d[i];
                vel[i] = momentum_ * vel[i] + gi;
                d[i] -= lr_ * vel[i];
            }
        }
    }
}

} // namespace tt
