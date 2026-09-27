// Gradient checks: every op's reverse-mode rule vs central finite differences.
#include <cmath>
#include <functional>
#include <gtest/gtest.h>

#include "tinytrain/ops.h"
#include "tinytrain/tensor.h"

namespace {

// Central differences on one input tensor; fwd() must rebuild the graph from
// the same handles on every call.
void gradcheck_one(const std::function<tt::Tensor()>& fwd, tt::Tensor& x,
                   double eps = 1e-3, double atol = 2e-3, double rtol = 2e-2) {
    int64_t n = x.numel();
    std::vector<double> numeric(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        float orig = x.data()[i];
        x.data()[i] = orig + static_cast<float>(eps);
        double lp = fwd().data()[0];
        x.data()[i] = orig - static_cast<float>(eps);
        double lm = fwd().data()[0];
        x.data()[i] = orig;
        numeric[static_cast<size_t>(i)] = (lp - lm) / (2 * eps);
    }
    ASSERT_TRUE(x.has_grad()) << "no gradient accumulated";
    for (int64_t i = 0; i < n; ++i) {
        double a = x.grad_data()[i];
        double e = numeric[static_cast<size_t>(i)];
        double tol = atol + rtol * std::fabs(e);
        EXPECT_LE(std::fabs(a - e), tol)
            << "mismatch at index " << i << ": analytic=" << a << " numeric=" << e;
    }
}

void gradcheck(const std::function<tt::Tensor()>& fwd,
               std::vector<tt::Tensor> inputs) {
    for (auto& t : inputs)
        t.zero_grad();
    tt::Tensor loss = fwd();
    ASSERT_EQ(loss.numel(), 1);
    loss.backward();
    for (auto& t : inputs)
        gradcheck_one(fwd, t);
}

tt::Tensor randn_grad(const std::vector<int64_t>& shape, uint64_t seed) {
    tt::Tensor t = tt::randn(shape, seed);
    t.set_requires_grad(true);
    return t;
}

} // namespace

TEST(Autograd, AddSubMulDiv) {
    auto a = randn_grad({2, 3}, 1);
    auto b = randn_grad({2, 3}, 2);
    gradcheck([&] { return tt::sum_all(tt::add(a, b)); }, {a, b});
    gradcheck([&] { return tt::sum_all(tt::sub(a, b)); }, {a, b});
    gradcheck([&] { return tt::sum_all(tt::mul(a, b)); }, {a, b});
    auto c = randn_grad({2, 3}, 3); // divisor, keep away from 0
    auto d = randn_grad({2, 3}, 4);
    gradcheck([&] { return tt::sum_all(tt::div(c, tt::add(d, tt::full({2, 3}, 2.0f)))); },
              {c, d});
}

TEST(Autograd, Broadcast) {
    auto a = randn_grad({2, 3}, 5);
    auto b = randn_grad({3}, 6); // broadcasts over dim 0
    gradcheck([&] { return tt::sum_all(tt::add(a, b)); }, {a, b});
    gradcheck([&] { return tt::sum_all(tt::mul(a, b)); }, {a, b});
    auto s = randn_grad({1}, 7); // scalar broadcast
    gradcheck([&] { return tt::sum_all(tt::mul(a, s)); }, {a, s});
}

TEST(Autograd, Unary) {
    auto a = randn_grad({2, 3}, 8);
    gradcheck([&] { return tt::sum_all(tt::exp(tt::div(a, tt::full({2, 3}, 4.0f)))); }, {a});
    gradcheck([&] { return tt::sum_all(tt::tanh(a)); }, {a});
    gradcheck([&] { return tt::sum_all(tt::silu(a)); }, {a});
    gradcheck([&] { return tt::sum_all(tt::sin(a)); }, {a});
    gradcheck([&] { return tt::sum_all(tt::cos(a)); }, {a});
    auto p = randn_grad({2, 3}, 9);
    gradcheck([&] { return tt::sum_all(tt::log(tt::add(tt::mul(p, p), tt::full({2, 3}, 1.0f)))); },
              {p});
    auto q = randn_grad({2, 3}, 10);
    gradcheck([&] { return tt::sum_all(tt::sqrt(tt::add(tt::mul(q, q), tt::full({2, 3}, 0.5f)))); },
              {q});
    auto r = randn_grad({2, 3}, 11);
    gradcheck([&] { return tt::sum_all(tt::pow(tt::add(r, tt::full({2, 3}, 2.0f)), 2.5f)); },
              {r});
    auto n = randn_grad({2, 3}, 12);
    gradcheck([&] { return tt::sum_all(tt::neg(n)); }, {n});
}

TEST(Autograd, Reductions) {
    auto a = randn_grad({2, 3, 4}, 13);
    gradcheck([&] { return tt::sum_all(tt::sum(a, 1)); }, {a});
    gradcheck([&] { return tt::sum_all(tt::sum(a, -1, true)); }, {a});
    gradcheck([&] { return tt::sum_all(tt::sum(a, 1, true)); }, {a});
    gradcheck([&] { return tt::sum_all(tt::sum(a, 0, true)); }, {a});
    gradcheck([&] { return tt::sum_all(a); }, {a});
    gradcheck([&] { return tt::mean(a); }, {a});
}

TEST(Autograd, ShapeOps) {
    auto a = randn_grad({2, 3, 4}, 14);
    gradcheck([&] { return tt::sum_all(tt::reshape(a, {6, 4})); }, {a});
    gradcheck([&] { return tt::sum_all(tt::transpose(a, 0, 2)); }, {a});
    gradcheck([&] { return tt::sum_all(tt::permute(a, {2, 0, 1})); }, {a});
    auto b = randn_grad({2, 1, 4}, 15);
    gradcheck([&] { return tt::sum_all(tt::expand(b, {2, 3, 4})); }, {b});
    auto c = randn_grad({2, 1, 4}, 16);
    gradcheck([&] { return tt::sum_all(tt::squeeze(c, 1)); }, {c});
    auto d = randn_grad({2, 4}, 17);
    gradcheck([&] { return tt::sum_all(tt::unsqueeze(d, 1)); }, {d});
}

TEST(Autograd, Matmul) {
    auto a = randn_grad({4, 5}, 18);
    auto b = randn_grad({5, 3}, 19);
    gradcheck([&] { return tt::sum_all(tt::matmul(a, b)); }, {a, b});
}

TEST(Autograd, Bmm) {
    auto a = randn_grad({2, 4, 5}, 20);
    auto b = randn_grad({2, 5, 3}, 21);
    gradcheck([&] { return tt::sum_all(tt::bmm(a, b)); }, {a, b});
}

TEST(Autograd, Embedding) {
    auto w = randn_grad({7, 4}, 22);
    std::vector<int64_t> idx = {0, 3, 3, 6, 1};
    gradcheck([&] { return tt::sum_all(tt::embedding(w, idx)); }, {w});
}

TEST(Autograd, Softmax) {
    auto a = randn_grad({2, 5}, 23);
    gradcheck([&] { return tt::sum_all(tt::softmax(a, -1)); }, {a});
    auto b = randn_grad({2, 3, 4}, 24);
    gradcheck([&] { return tt::sum_all(tt::softmax(b, 1)); }, {b});
}

TEST(Autograd, CrossEntropy) {
    auto logits = randn_grad({6, 5}, 25);
    std::vector<int64_t> targets = {0, 4, 2, 1, 3, 2};
    gradcheck([&] { return tt::cross_entropy(logits, targets); }, {logits});
}

TEST(Autograd, Rmsnorm) {
    auto a = randn_grad({2, 3, 8}, 26);
    auto w = randn_grad({8}, 27);
    gradcheck([&] { return tt::sum_all(tt::rmsnorm(a, w, 1e-5f)); }, {a, w});
}

TEST(Autograd, Dropout) {
    auto a = randn_grad({2, 8}, 28);
    // Fixed seed -> deterministic mask; gradcheck perturbs values only.
    gradcheck([&] { return tt::sum_all(tt::dropout(a, 0.5f, 42)); }, {a});
}

TEST(Autograd, Rope) {
    auto x = randn_grad({2, 5, 3, 8}, 29);
    // cos/sin tables as constants (the realistic use)
    tt::Tensor cos = tt::rand_uniform({5, 4}, -1.0f, 1.0f, 30);
    tt::Tensor sin = tt::rand_uniform({5, 4}, -1.0f, 1.0f, 31);
    gradcheck([&] { return tt::sum_all(tt::rope(x, cos, sin)); }, {x});
}

TEST(Autograd, GradAccumulates) {
    auto a = randn_grad({2, 2}, 32);
    auto b = tt::mul(a, a); // a^2
    tt::sum_all(b).backward();
    float g1 = a.grad_data()[0];
    tt::sum_all(b).backward(); // second backward on a fresh graph
    EXPECT_FLOAT_EQ(a.grad_data()[0], 2 * g1);
}

TEST(Autograd, LinearRegressionEndToEnd) {
    // y = 3x + 1 recovered by manual gradient descent: the whole engine,
    // wired together, actually learns.
    tt::Tensor w = tt::randn({1}, 100);
    tt::Tensor b = tt::zeros({1});
    w.set_requires_grad(true);
    b.set_requires_grad(true);
    tt::Tensor x = tt::full({16, 1}, 0.0f);
    for (int64_t i = 0; i < 16; ++i)
        x.data()[i] = -2.0f + i * (4.0f / 15);
    for (int step = 0; step < 300; ++step) {
        w.zero_grad();
        b.zero_grad();
        // pred = x*w + b, mse = mean((pred - y)^2)
        tt::Tensor pred = tt::add(tt::mul(x, tt::expand(w, {16, 1})), tt::expand(b, {16, 1}));
        tt::Tensor y = tt::add(tt::mul(x, tt::full({16, 1}, 3.0f)), tt::full({16, 1}, 1.0f));
        tt::Tensor diff = tt::sub(pred, y);
        tt::Tensor loss = tt::mean(tt::mul(diff, diff));
        loss.backward();
        float lr = 0.1f;
        w.data()[0] -= lr * w.grad_data()[0];
        b.data()[0] -= lr * b.grad_data()[0];
    }
    EXPECT_NEAR(w.data()[0], 3.0, 0.05);
    EXPECT_NEAR(b.data()[0], 1.0, 0.05);
}
