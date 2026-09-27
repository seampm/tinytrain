#include "tinytrain/ops.h"

#include <cmath>
#include <stdexcept>

namespace tt {

namespace {

// ---------- small utilities ----------

void ensure_grad(const std::shared_ptr<Tensor::Impl>& impl) {
    if (impl->grad.empty())
        impl->grad.assign(static_cast<size_t>(impl->numel), 0.0f);
}

std::vector<int64_t> broadcast_shape(const std::vector<int64_t>& a,
                                     const std::vector<int64_t>& b) {
    int64_t ar = static_cast<int64_t>(a.size()), br = static_cast<int64_t>(b.size());
    int64_t r = std::max(ar, br);
    std::vector<int64_t> out(static_cast<size_t>(r));
    for (int64_t i = 0; i < r; ++i) {
        int64_t ad = (i < r - ar) ? 1 : a[static_cast<size_t>(i - (r - ar))];
        int64_t bd = (i < r - br) ? 1 : b[static_cast<size_t>(i - (r - br))];
        if (ad != bd && ad != 1 && bd != 1)
            throw std::invalid_argument("broadcast: incompatible shapes");
        out[static_cast<size_t>(i)] = std::max(ad, bd);
    }
    return out;
}

// dst->grad[doff] += src[l] for every src element, folding src's shape onto
// dst's by summing over broadcast dims. src must broadcast to dst's shape.
void bcast_add(const std::shared_ptr<Tensor::Impl>& dst, const float* src,
               const std::vector<int64_t>& src_shape, const std::vector<int64_t>& src_strides,
               const std::vector<int64_t>& dst_shape, const std::vector<int64_t>& dst_strides) {
    if (!dst->requires_grad)
        return;
    ensure_grad(dst);
    int64_t srank = static_cast<int64_t>(src_shape.size());
    int64_t drank = static_cast<int64_t>(dst_shape.size());
    int64_t snumel = 1;
    for (int64_t d : src_shape)
        snumel *= d;
    std::vector<int64_t> idx(static_cast<size_t>(srank));
    for (int64_t l = 0; l < snumel; ++l) {
        int64_t rem = l;
        for (int64_t d = 0; d < srank; ++d) {
            idx[static_cast<size_t>(d)] = rem / src_strides[static_cast<size_t>(d)];
            rem %= src_strides[static_cast<size_t>(d)];
        }
        int64_t doff = 0;
        for (int64_t d = 0; d < drank; ++d) {
            int64_t sd = srank - drank + d; // trailing alignment
            int64_t ii = (dst_shape[static_cast<size_t>(d)] == 1)
                             ? 0
                             : idx[static_cast<size_t>(sd)];
            doff += ii * dst_strides[static_cast<size_t>(d)];
        }
        dst->grad[static_cast<size_t>(doff)] += src[l];
    }
}

// Read one broadcast element: out multi-index -> input offset.
float bcast_get(const float* data, const std::vector<int64_t>& shape,
                const std::vector<int64_t>& strides, const std::vector<int64_t>& idx) {
    int64_t orank = static_cast<int64_t>(idx.size());
    int64_t drank = static_cast<int64_t>(shape.size());
    int64_t off = 0;
    for (int64_t d = 0; d < drank; ++d) {
        int64_t ii = (shape[static_cast<size_t>(d)] == 1)
                         ? 0
                         : idx[static_cast<size_t>(orank - drank + d)];
        off += ii * strides[static_cast<size_t>(d)];
    }
    return data[off];
}

Tensor make_out(std::vector<int64_t> shape, bool requires_grad) {
    return Tensor(std::move(shape), requires_grad);
}

int64_t norm_dim(int64_t dim, int64_t rank) {
    if (dim < 0)
        dim += rank;
    if (dim < 0 || dim >= rank)
        throw std::invalid_argument("dim out of range");
    return dim;
}

// C[M,N] = A[M,K] @ B[K,N], row-major contiguous.
//
// Cache-blocked (MB x NB tiles): each thread's C tile (MB*NB) stays in L2
// across the K loop while the B panel row (NB) is reused from L1 across the
// MB rows. This keeps the big B matrix from streaming once per C row.
void sgemm(const float* A, const float* B, float* C, int64_t M, int64_t K, int64_t N) {
    constexpr int64_t MB = 64;
    constexpr int64_t NB = 2048;
    int64_t mblocks = (M + MB - 1) / MB;
    int64_t nblocks = (N + NB - 1) / NB;
#pragma omp parallel for collapse(2) schedule(static)
    for (int64_t nb = 0; nb < nblocks; ++nb) {
        for (int64_t mb = 0; mb < mblocks; ++mb) {
            int64_t j0 = nb * NB;
            int64_t j1 = j0 + NB < N ? j0 + NB : N;
            int64_t i0 = mb * MB;
            int64_t i1 = i0 + MB < M ? i0 + MB : M;
            int64_t nn = j1 - j0;
            for (int64_t i = i0; i < i1; ++i) {
                float* crow = C + i * N + j0;
                for (int64_t j = 0; j < nn; ++j)
                    crow[j] = 0.0f;
            }
            for (int64_t k = 0; k < K; ++k) {
                const float* brow = B + k * N + j0;
                for (int64_t i = i0; i < i1; ++i) {
                    float aik = A[i * K + k];
                    float* crow = C + i * N + j0;
                    for (int64_t j = 0; j < nn; ++j)
                        crow[j] += aik * brow[j];
                }
            }
        }
    }
}

void transpose_copy(const float* src, float* dst, int64_t rows, int64_t cols) {
    for (int64_t i = 0; i < rows; ++i)
        for (int64_t j = 0; j < cols; ++j)
            dst[j * rows + i] = src[i * cols + j];
}

} // namespace

// ---------- elementwise ----------

namespace {

Tensor elem_binary(const Tensor& a, const Tensor& b,
                   float (*fwd)(float, float),
                   void (*bwd)(float, float, float, float&, float&)) {
    auto shape = broadcast_shape(a.shape(), b.shape());
    Tensor out = make_out(shape, a.requires_grad() || b.requires_grad());
    int64_t n = out.numel();
    int64_t rank = static_cast<int64_t>(shape.size());
    std::vector<int64_t> ostrides(static_cast<size_t>(rank));
    int64_t s = 1;
    for (int64_t d = rank - 1; d >= 0; --d) {
        ostrides[static_cast<size_t>(d)] = s;
        s *= shape[static_cast<size_t>(d)];
    }
    std::vector<int64_t> idx(static_cast<size_t>(rank));
    const float* ad = a.data();
    const float* bd = b.data();
    float* od = out.data();
    const auto& as = a.shape(), & bs = b.shape();
    const auto& ast = a.impl()->strides, & bst = b.impl()->strides;
    for (int64_t l = 0; l < n; ++l) {
        int64_t rem = l;
        for (int64_t d = 0; d < rank; ++d) {
            idx[static_cast<size_t>(d)] = rem / ostrides[static_cast<size_t>(d)];
            rem %= ostrides[static_cast<size_t>(d)];
        }
        float av = bcast_get(ad, as, ast, idx);
        float bv = bcast_get(bd, bs, bst, idx);
        od[l] = fwd(av, bv);
    }
    if (out.requires_grad()) {
        auto ao = a.impl(), bo = b.impl(), oo = out.impl();
        oo->prev = {ao, bo};
        oo->backward_fn = [ao, bo, oo, oshape = shape, ostrides, fwd, bwd]() {
            const float* g = oo->grad.data();
            int64_t n = oo->numel;
            int64_t rank = static_cast<int64_t>(oshape.size());
            std::vector<int64_t> idx(static_cast<size_t>(rank));
            const float* ad = ao->data.data();
            const float* bd = bo->data.data();
            const auto& as = ao->shape, & bs = bo->shape;
            const auto& ast = ao->strides, & bst = bo->strides;
            // accumulate per-input contributions; then bcast_add folds them in.
            // To avoid temporaries, accumulate directly with index mapping.
            if (ao->requires_grad) {
                ensure_grad(ao);
                for (int64_t l = 0; l < n; ++l) {
                    int64_t rem = l;
                    for (int64_t d = 0; d < rank; ++d) {
                        idx[static_cast<size_t>(d)] = rem / ostrides[static_cast<size_t>(d)];
                        rem %= ostrides[static_cast<size_t>(d)];
                    }
                    float av = bcast_get(ad, as, ast, idx);
                    float bv = bcast_get(bd, bs, bst, idx);
                    float ga = 0, gb = 0;
                    bwd(av, bv, g[l], ga, gb);
                    if (ga != 0.0f) {
                        int64_t orank = rank, drank = static_cast<int64_t>(as.size());
                        int64_t off = 0;
                        for (int64_t d = 0; d < drank; ++d) {
                            int64_t ii = (as[static_cast<size_t>(d)] == 1)
                                             ? 0
                                             : idx[static_cast<size_t>(orank - drank + d)];
                            off += ii * ast[static_cast<size_t>(d)];
                        }
                        ao->grad[static_cast<size_t>(off)] += ga;
                    }
                    if (gb != 0.0f && bo->requires_grad) {
                        int64_t orank = rank, drank = static_cast<int64_t>(bs.size());
                        int64_t off = 0;
                        for (int64_t d = 0; d < drank; ++d) {
                            int64_t ii = (bs[static_cast<size_t>(d)] == 1)
                                             ? 0
                                             : idx[static_cast<size_t>(orank - drank + d)];
                            off += ii * bst[static_cast<size_t>(d)];
                        }
                        ensure_grad(bo);
                        bo->grad[static_cast<size_t>(off)] += gb;
                    }
                }
            } else if (bo->requires_grad) {
                ensure_grad(bo);
                for (int64_t l = 0; l < n; ++l) {
                    int64_t rem = l;
                    for (int64_t d = 0; d < rank; ++d) {
                        idx[static_cast<size_t>(d)] = rem / ostrides[static_cast<size_t>(d)];
                        rem %= ostrides[static_cast<size_t>(d)];
                    }
                    float av = bcast_get(ad, as, ast, idx);
                    float bv = bcast_get(bd, bs, bst, idx);
                    float ga = 0, gb = 0;
                    bwd(av, bv, g[l], ga, gb);
                    if (gb != 0.0f) {
                        int64_t orank = rank, drank = static_cast<int64_t>(bs.size());
                        int64_t off = 0;
                        for (int64_t d = 0; d < drank; ++d) {
                            int64_t ii = (bs[static_cast<size_t>(d)] == 1)
                                             ? 0
                                             : idx[static_cast<size_t>(orank - drank + d)];
                            off += ii * bst[static_cast<size_t>(d)];
                        }
                        bo->grad[static_cast<size_t>(off)] += gb;
                    }
                }
            }
            (void)fwd;
        };
    }
    return out;
}

float fwd_add(float a, float b) { return a + b; }
float fwd_sub(float a, float b) { return a - b; }
float fwd_mul(float a, float b) { return a * b; }
float fwd_div(float a, float b) { return a / b; }
void bwd_add(float, float, float g, float& ga, float& gb) { ga = g; gb = g; }
void bwd_sub(float, float, float g, float& ga, float& gb) { ga = g; gb = -g; }
void bwd_mul(float a, float b, float g, float& ga, float& gb) { ga = g * b; gb = g * a; }
void bwd_div(float a, float b, float g, float& ga, float& gb) {
    ga = g / b;
    gb = -g * a / (b * b);
}

} // namespace

Tensor add(const Tensor& a, const Tensor& b) { return elem_binary(a, b, fwd_add, bwd_add); }
Tensor sub(const Tensor& a, const Tensor& b) { return elem_binary(a, b, fwd_sub, bwd_sub); }
Tensor mul(const Tensor& a, const Tensor& b) { return elem_binary(a, b, fwd_mul, bwd_mul); }
Tensor div(const Tensor& a, const Tensor& b) { return elem_binary(a, b, fwd_div, bwd_div); }

Tensor neg(const Tensor& a) {
    Tensor out = make_out(a.shape(), a.requires_grad());
    int64_t n = a.numel();
    for (int64_t i = 0; i < n; ++i)
        out.data()[i] = -a.data()[i];
    if (out.requires_grad()) {
        auto ao = a.impl(), oo = out.impl();
        oo->prev = {ao};
        oo->backward_fn = [ao, oo]() {
            if (!ao->requires_grad)
                return;
            ensure_grad(ao);
            const float* g = oo->grad.data();
            for (int64_t i = 0; i < oo->numel; ++i)
                ao->grad[static_cast<size_t>(i)] -= g[i];
        };
    }
    return out;
}

namespace {

Tensor elem_unary(const Tensor& a, float (*fwd)(float), float (*bwd)(float)) {
    Tensor out = make_out(a.shape(), a.requires_grad());
    int64_t n = a.numel();
    const float* ad = a.data();
    float* od = out.data();
    for (int64_t i = 0; i < n; ++i)
        od[i] = fwd(ad[i]);
    if (out.requires_grad()) {
        auto ao = a.impl(), oo = out.impl();
        oo->prev = {ao};
        oo->backward_fn = [ao, oo, bwd]() {
            if (!ao->requires_grad)
                return;
            ensure_grad(ao);
            const float* g = oo->grad.data();
            const float* ad = ao->data.data();
            for (int64_t i = 0; i < oo->numel; ++i)
                ao->grad[static_cast<size_t>(i)] += g[i] * bwd(ad[i]);
        };
    }
    return out;
}

float f_exp(float x) { return std::exp(x); }
float b_exp(float x) { return std::exp(x); }
float f_log(float x) { return std::log(x); }
float b_log(float x) { return 1.0f / x; }
float f_sqrt(float x) { return std::sqrt(x); }
float b_sqrt(float x) { return 0.5f / std::sqrt(x); }
float f_tanh(float x) { return std::tanh(x); }
float b_tanh(float x) {
    float t = std::tanh(x);
    return 1.0f - t * t;
}
float f_silu(float x) { return x / (1.0f + std::exp(-x)); }
float b_silu(float x) {
    float s = 1.0f / (1.0f + std::exp(-x));
    return s * (1.0f + x * (1.0f - s));
}
float f_sin(float x) { return std::sin(x); }
float b_sin(float x) { return std::cos(x); }
float f_cos(float x) { return std::cos(x); }
float b_cos(float x) { return -std::sin(x); }

} // namespace

Tensor exp(const Tensor& a) { return elem_unary(a, f_exp, b_exp); }
Tensor log(const Tensor& a) { return elem_unary(a, f_log, b_log); }
Tensor sqrt(const Tensor& a) { return elem_unary(a, f_sqrt, b_sqrt); }
Tensor tanh(const Tensor& a) { return elem_unary(a, f_tanh, b_tanh); }
Tensor silu(const Tensor& a) { return elem_unary(a, f_silu, b_silu); }
Tensor sin(const Tensor& a) { return elem_unary(a, f_sin, b_sin); }
Tensor cos(const Tensor& a) { return elem_unary(a, f_cos, b_cos); }

Tensor pow(const Tensor& a, float p) {
    Tensor out = make_out(a.shape(), a.requires_grad());
    int64_t n = a.numel();
    for (int64_t i = 0; i < n; ++i)
        out.data()[i] = std::pow(a.data()[i], p);
    if (out.requires_grad()) {
        auto ao = a.impl(), oo = out.impl();
        oo->prev = {ao};
        oo->backward_fn = [ao, oo, p]() {
            if (!ao->requires_grad)
                return;
            ensure_grad(ao);
            const float* g = oo->grad.data();
            const float* ad = ao->data.data();
            for (int64_t i = 0; i < oo->numel; ++i)
                ao->grad[static_cast<size_t>(i)] += g[i] * p * std::pow(ad[i], p - 1.0f);
        };
    }
    return out;
}

// ---------- reductions ----------

Tensor sum(const Tensor& a, int64_t dim, bool keepdim) {
    dim = norm_dim(dim, a.ndim());
    std::vector<int64_t> oshape;
    for (int64_t d = 0; d < a.ndim(); ++d) {
        if (d == dim) {
            if (keepdim)
                oshape.push_back(1);
        } else {
            oshape.push_back(a.dim(d));
        }
    }
    if (oshape.empty())
        oshape.push_back(1); // sum over the only dim -> scalar-ish {1}
    Tensor out = make_out(oshape, a.requires_grad());
    int64_t n = out.numel();
    for (int64_t i = 0; i < n; ++i)
        out.data()[i] = 0.0f;
    // accumulate
    int64_t in_rank = a.ndim();
    std::vector<int64_t> idx(static_cast<size_t>(in_rank));
    const auto& istr = a.impl()->strides;
    const auto& ostr = out.impl()->strides;
    int64_t orank = static_cast<int64_t>(oshape.size());
    for (int64_t l = 0; l < a.numel(); ++l) {
        int64_t rem = l;
        for (int64_t d = 0; d < in_rank; ++d) {
            idx[static_cast<size_t>(d)] = rem / istr[static_cast<size_t>(d)];
            rem %= istr[static_cast<size_t>(d)];
        }
        int64_t ooff = 0;
        for (int64_t d = 0, od = 0; d < in_rank; ++d) {
            if (d == dim) {
                if (keepdim)
                    ooff += 0 * ostr[static_cast<size_t>(od++)];
                continue;
            }
            ooff += idx[static_cast<size_t>(d)] * ostr[static_cast<size_t>(od++)];
        }
        (void)orank;
        out.data()[ooff] += a.data()[l];
    }
    if (out.requires_grad()) {
        auto ao = a.impl(), oo = out.impl();
        oo->prev = {ao};
        oo->backward_fn = [ao, oo, dim, keepdim, in_shape = a.shape(), in_strides = a.impl()->strides,
                           out_strides = out.impl()->strides]() {
            if (!ao->requires_grad)
                return;
            ensure_grad(ao);
            const float* g = oo->grad.data();
            int64_t in_rank = static_cast<int64_t>(in_shape.size());
            std::vector<int64_t> idx(static_cast<size_t>(in_rank));
            for (int64_t l = 0; l < ao->numel; ++l) {
                int64_t rem = l;
                for (int64_t d = 0; d < in_rank; ++d) {
                    idx[static_cast<size_t>(d)] = rem / in_strides[static_cast<size_t>(d)];
                    rem %= in_strides[static_cast<size_t>(d)];
                }
                int64_t ooff = 0;
                for (int64_t d = 0, od = 0; d < in_rank; ++d) {
                    if (d == dim) {
                        if (keepdim)
                            od++; // keepdim retains the size-1 dim in the output
                        continue;
                    }
                    ooff += idx[static_cast<size_t>(d)] * out_strides[static_cast<size_t>(od++)];
                }
                ao->grad[static_cast<size_t>(l)] += g[ooff];
            }
        };
    }
    return out;
}

Tensor sum_all(const Tensor& a) {
    Tensor out = make_out({1}, a.requires_grad());
    double acc = 0;
    for (int64_t i = 0; i < a.numel(); ++i)
        acc += a.data()[i];
    out.data()[0] = static_cast<float>(acc);
    if (out.requires_grad()) {
        auto ao = a.impl(), oo = out.impl();
        oo->prev = {ao};
        oo->backward_fn = [ao, oo]() {
            if (!ao->requires_grad)
                return;
            ensure_grad(ao);
            float g = oo->grad[0];
            for (int64_t i = 0; i < ao->numel; ++i)
                ao->grad[static_cast<size_t>(i)] += g;
        };
    }
    return out;
}

Tensor mean(const Tensor& a) {
    Tensor s = sum_all(a);
    Tensor out = make_out({1}, s.requires_grad());
    out.data()[0] = s.data()[0] / static_cast<float>(a.numel());
    if (out.requires_grad()) {
        auto so = s.impl(), oo = out.impl();
        oo->prev = {so};
        // d out / d s = 1 / a.numel ; chain into s's own backward
        int64_t an = a.numel();
        oo->backward_fn = [so, oo, an]() {
            if (!so->requires_grad)
                return;
            ensure_grad(so);
            so->grad[0] += oo->grad[0] / static_cast<float>(an);
        };
    }
    return out;
}

// ---------- shape ops ----------

Tensor reshape(const Tensor& a, const std::vector<int64_t>& shape) {
    int64_t n = 1;
    int64_t infer = -1;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == -1) {
            if (infer != -1)
                throw std::invalid_argument("reshape: only one -1 allowed");
            infer = static_cast<int64_t>(i);
        } else {
            if (shape[i] <= 0)
                throw std::invalid_argument("reshape: dims must be positive");
            n *= shape[i];
        }
    }
    std::vector<int64_t> out_shape = shape;
    if (infer != -1) {
        if (a.numel() % n != 0)
            throw std::invalid_argument("reshape: incompatible size");
        out_shape[static_cast<size_t>(infer)] = a.numel() / n;
        n = a.numel();
    }
    if (n != a.numel())
        throw std::invalid_argument("reshape: incompatible size");
    Tensor out = make_out(out_shape, a.requires_grad());
    for (int64_t i = 0; i < n; ++i)
        out.data()[i] = a.data()[i];
    if (out.requires_grad()) {
        auto ao = a.impl(), oo = out.impl();
        auto ashape = a.shape();
        oo->prev = {ao};
        oo->backward_fn = [ao, oo, ashape]() {
            if (!ao->requires_grad)
                return;
            ensure_grad(ao);
            for (int64_t i = 0; i < oo->numel; ++i)
                ao->grad[static_cast<size_t>(i)] += oo->grad[static_cast<size_t>(i)];
        };
    }
    return out;
}

Tensor transpose(const Tensor& a, int64_t d0, int64_t d1) {
    d0 = norm_dim(d0, a.ndim());
    d1 = norm_dim(d1, a.ndim());
    std::vector<int64_t> dims(static_cast<size_t>(a.ndim()));
    for (int64_t d = 0; d < a.ndim(); ++d)
        dims[static_cast<size_t>(d)] = d;
    std::swap(dims[static_cast<size_t>(d0)], dims[static_cast<size_t>(d1)]);
    return permute(a, dims);
}

Tensor permute(const Tensor& a, const std::vector<int64_t>& dims) {
    int64_t rank = a.ndim();
    if (static_cast<int64_t>(dims.size()) != rank)
        throw std::invalid_argument("permute: rank mismatch");
    std::vector<int64_t> oshape(static_cast<size_t>(rank));
    for (int64_t d = 0; d < rank; ++d) {
        int64_t s = norm_dim(dims[static_cast<size_t>(d)], rank);
        oshape[static_cast<size_t>(d)] = a.dim(s);
    }
    Tensor out = make_out(oshape, a.requires_grad());
    // out[idx] = a[idx permuted back]: out multi-index i_d over oshape,
    // input index along original dim k is i_{position of k in dims}.
    std::vector<int64_t> inv(static_cast<size_t>(rank));
    for (int64_t d = 0; d < rank; ++d)
        inv[static_cast<size_t>(norm_dim(dims[static_cast<size_t>(d)], rank))] = d;
    const auto& istr = a.impl()->strides;
    const auto& ostr = out.impl()->strides;
    std::vector<int64_t> idx(static_cast<size_t>(rank));
    for (int64_t l = 0; l < out.numel(); ++l) {
        int64_t rem = l;
        for (int64_t d = 0; d < rank; ++d) {
            idx[static_cast<size_t>(d)] = rem / ostr[static_cast<size_t>(d)];
            rem %= ostr[static_cast<size_t>(d)];
        }
        int64_t ioff = 0;
        for (int64_t k = 0; k < rank; ++k)
            ioff += idx[static_cast<size_t>(inv[static_cast<size_t>(k)])] * istr[static_cast<size_t>(k)];
        out.data()[l] = a.data()[ioff];
    }
    if (out.requires_grad()) {
        auto ao = a.impl(), oo = out.impl();
        oo->prev = {ao};
        // backward of permute(dims) is permute(inv(dims))
        oo->backward_fn = [ao, oo, inv]() {
            if (!ao->requires_grad)
                return;
            ensure_grad(ao);
            int64_t rank = static_cast<int64_t>(oo->shape.size());
            const auto& gstr = oo->strides;
            const auto& astr = ao->strides;
            std::vector<int64_t> idx(static_cast<size_t>(rank));
            // grad_out has out's shape; permute it back with inv
            for (int64_t l = 0; l < oo->numel; ++l) {
                int64_t rem = l;
                for (int64_t d = 0; d < rank; ++d) {
                    idx[static_cast<size_t>(d)] = rem / gstr[static_cast<size_t>(d)];
                    rem %= gstr[static_cast<size_t>(d)];
                }
                int64_t aoff = 0;
                for (int64_t k = 0; k < rank; ++k)
                    aoff += idx[static_cast<size_t>(inv[static_cast<size_t>(k)])] * astr[static_cast<size_t>(k)];
                ao->grad[static_cast<size_t>(aoff)] += oo->grad[static_cast<size_t>(l)];
            }
        };
    }
    return out;
}

Tensor expand(const Tensor& a, const std::vector<int64_t>& shape) {
    // verify broadcastable, then materialize
    (void)broadcast_shape(a.shape(), shape);
    // stricter: every dim of a must be 1 or equal
    int64_t ar = a.ndim(), r = static_cast<int64_t>(shape.size());
    for (int64_t i = 0; i < r; ++i) {
        int64_t ad = (i < r - ar) ? 1 : a.dim(i - (r - ar));
        if (ad != 1 && ad != shape[static_cast<size_t>(i)])
            throw std::invalid_argument("expand: incompatible shape");
    }
    Tensor out = make_out(shape, a.requires_grad());
    int64_t rank = r;
    std::vector<int64_t> ostrides(static_cast<size_t>(rank));
    int64_t s = 1;
    for (int64_t d = rank - 1; d >= 0; --d) {
        ostrides[static_cast<size_t>(d)] = s;
        s *= shape[static_cast<size_t>(d)];
    }
    std::vector<int64_t> idx(static_cast<size_t>(rank));
    for (int64_t l = 0; l < out.numel(); ++l) {
        int64_t rem = l;
        for (int64_t d = 0; d < rank; ++d) {
            idx[static_cast<size_t>(d)] = rem / ostrides[static_cast<size_t>(d)];
            rem %= ostrides[static_cast<size_t>(d)];
        }
        out.data()[l] = bcast_get(a.data(), a.shape(), a.impl()->strides, idx);
    }
    if (out.requires_grad()) {
        auto ao = a.impl(), oo = out.impl();
        auto oshape = shape;
        auto ostrides_c = ostrides;
        oo->prev = {ao};
        oo->backward_fn = [ao, oo, oshape, ostrides_c]() {
            bcast_add(ao, oo->grad.data(), oshape, ostrides_c, ao->shape, ao->strides);
        };
    }
    return out;
}

Tensor squeeze(const Tensor& a, int64_t dim) {
    dim = norm_dim(dim, a.ndim());
    if (a.dim(dim) != 1)
        throw std::invalid_argument("squeeze: dim size != 1");
    std::vector<int64_t> oshape;
    for (int64_t d = 0; d < a.ndim(); ++d)
        if (d != dim)
            oshape.push_back(a.dim(d));
    return reshape(a, oshape);
}

Tensor unsqueeze(const Tensor& a, int64_t dim) {
    int64_t rank = a.ndim() + 1;
    if (dim < 0)
        dim += rank;
    if (dim < 0 || dim > a.ndim())
        throw std::invalid_argument("unsqueeze: dim out of range");
    std::vector<int64_t> oshape;
    for (int64_t d = 0; d < a.ndim(); ++d) {
        if (d == dim)
            oshape.push_back(1);
        oshape.push_back(a.dim(d));
    }
    if (dim == a.ndim())
        oshape.push_back(1);
    return reshape(a, oshape);
}

// ---------- linear algebra ----------

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.ndim() != 2 || b.ndim() != 2)
        throw std::invalid_argument("matmul: both inputs must be 2D");
    int64_t M = a.dim(0), K = a.dim(1), K2 = b.dim(0), N = b.dim(1);
    if (K != K2)
        throw std::invalid_argument("matmul: inner dims mismatch");
    Tensor out = make_out({M, N}, a.requires_grad() || b.requires_grad());
    sgemm(a.data(), b.data(), out.data(), M, K, N);
    if (out.requires_grad()) {
        auto ao = a.impl(), bo = b.impl(), oo = out.impl();
        oo->prev = {ao, bo};
        oo->backward_fn = [ao, bo, oo, M, K, N]() {
            const float* g = oo->grad.data();
            if (ao->requires_grad) {
                ensure_grad(ao);
                // gA = g @ B^T
                std::vector<float> bt(static_cast<size_t>(K * N));
                transpose_copy(bo->data.data(), bt.data(), K, N);
                std::vector<float> ga(static_cast<size_t>(M * K));
                sgemm(g, bt.data(), ga.data(), M, N, K);
                for (int64_t i = 0; i < M * K; ++i)
                    ao->grad[static_cast<size_t>(i)] += ga[static_cast<size_t>(i)];
            }
            if (bo->requires_grad) {
                ensure_grad(bo);
                // gB = A^T @ g
                std::vector<float> at(static_cast<size_t>(M * K));
                transpose_copy(ao->data.data(), at.data(), M, K);
                std::vector<float> gb(static_cast<size_t>(K * N));
                sgemm(at.data(), g, gb.data(), K, M, N);
                for (int64_t i = 0; i < K * N; ++i)
                    bo->grad[static_cast<size_t>(i)] += gb[static_cast<size_t>(i)];
            }
        };
    }
    return out;
}

Tensor bmm(const Tensor& a, const Tensor& b) {
    if (a.ndim() != 3 || b.ndim() != 3)
        throw std::invalid_argument("bmm: both inputs must be 3D");
    int64_t B = a.dim(0), M = a.dim(1), K = a.dim(2);
    if (b.dim(0) != B || b.dim(1) != K)
        throw std::invalid_argument("bmm: batch/inner dims mismatch");
    int64_t N = b.dim(2);
    Tensor out = make_out({B, M, N}, a.requires_grad() || b.requires_grad());
    for (int64_t bidx = 0; bidx < B; ++bidx)
        sgemm(a.data() + bidx * M * K, b.data() + bidx * K * N,
              out.data() + bidx * M * N, M, K, N);
    if (out.requires_grad()) {
        auto ao = a.impl(), bo = b.impl(), oo = out.impl();
        oo->prev = {ao, bo};
        oo->backward_fn = [ao, bo, oo, B, M, K, N]() {
            const float* g = oo->grad.data();
            if (ao->requires_grad) {
                ensure_grad(ao);
                std::vector<float> bt(static_cast<size_t>(K * N));
                std::vector<float> ga(static_cast<size_t>(M * K));
                for (int64_t bidx = 0; bidx < B; ++bidx) {
                    transpose_copy(bo->data.data() + bidx * K * N, bt.data(), K, N);
                    sgemm(g + bidx * M * N, bt.data(), ga.data(), M, N, K);
                    float* dst = ao->grad.data() + bidx * M * K;
                    for (int64_t i = 0; i < M * K; ++i)
                        dst[i] += ga[static_cast<size_t>(i)];
                }
            }
            if (bo->requires_grad) {
                ensure_grad(bo);
                std::vector<float> at(static_cast<size_t>(M * K));
                std::vector<float> gb(static_cast<size_t>(K * N));
                for (int64_t bidx = 0; bidx < B; ++bidx) {
                    transpose_copy(ao->data.data() + bidx * M * K, at.data(), M, K);
                    sgemm(at.data(), g + bidx * M * N, gb.data(), K, M, N);
                    float* dst = bo->grad.data() + bidx * K * N;
                    for (int64_t i = 0; i < K * N; ++i)
                        dst[i] += gb[static_cast<size_t>(i)];
                }
            }
        };
    }
    return out;
}

// ---------- neural nets ----------

Tensor embedding(const Tensor& weight, const std::vector<int64_t>& idx) {
    int64_t V = weight.dim(0), D = weight.dim(1);
    int64_t N = static_cast<int64_t>(idx.size());
    Tensor out = make_out({N, D}, weight.requires_grad());
    for (int64_t i = 0; i < N; ++i) {
        if (idx[static_cast<size_t>(i)] < 0 || idx[static_cast<size_t>(i)] >= V)
            throw std::invalid_argument("embedding: index out of range");
        const float* row = weight.data() + idx[static_cast<size_t>(i)] * D;
        float* orow = out.data() + i * D;
        for (int64_t j = 0; j < D; ++j)
            orow[j] = row[j];
    }
    if (out.requires_grad()) {
        auto wo = weight.impl(), oo = out.impl();
        oo->prev = {wo};
        oo->backward_fn = [wo, oo, idx, D]() {
            if (!wo->requires_grad)
                return;
            ensure_grad(wo);
            const float* g = oo->grad.data();
            for (size_t i = 0; i < idx.size(); ++i) {
                float* grow = wo->grad.data() + idx[i] * D;
                const float* grow_in = g + i * D;
                for (int64_t j = 0; j < D; ++j)
                    grow[j] += grow_in[j];
            }
        };
    }
    return out;
}

Tensor softmax(const Tensor& a, int64_t dim) {
    dim = norm_dim(dim, a.ndim());
    Tensor out = make_out(a.shape(), a.requires_grad());
    // reduce over dim: iterate outer/inner
    int64_t outer = 1, inner = 1, dsize = a.dim(dim);
    for (int64_t d = 0; d < dim; ++d)
        outer *= a.dim(d);
    for (int64_t d = dim + 1; d < a.ndim(); ++d)
        inner *= a.dim(d);
    const float* ad = a.data();
    float* od = out.data();
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t j = 0; j < inner; ++j) {
            float m = -1e30f;
            for (int64_t k = 0; k < dsize; ++k) {
                float v = ad[(o * dsize + k) * inner + j];
                if (v > m)
                    m = v;
            }
            double s = 0;
            for (int64_t k = 0; k < dsize; ++k) {
                float e = std::exp(ad[(o * dsize + k) * inner + j] - m);
                od[(o * dsize + k) * inner + j] = e;
                s += e;
            }
            float inv = static_cast<float>(1.0 / s);
            for (int64_t k = 0; k < dsize; ++k)
                od[(o * dsize + k) * inner + j] *= inv;
        }
    }
    if (out.requires_grad()) {
        auto ao = a.impl(), oo = out.impl();
        oo->prev = {ao};
        oo->backward_fn = [ao, oo, dim]() {
            if (!ao->requires_grad)
                return;
            ensure_grad(ao);
            int64_t outer = 1, inner = 1, dsize = ao->shape[static_cast<size_t>(dim)];
            for (int64_t d = 0; d < dim; ++d)
                outer *= ao->shape[static_cast<size_t>(d)];
            for (int64_t d = dim + 1; d < static_cast<int64_t>(ao->shape.size()); ++d)
                inner *= ao->shape[static_cast<size_t>(d)];
            const float* s = oo->data.data();
            const float* g = oo->grad.data();
            float* ag = ao->grad.data();
            for (int64_t o = 0; o < outer; ++o) {
                for (int64_t j = 0; j < inner; ++j) {
                    double dot = 0;
                    for (int64_t k = 0; k < dsize; ++k) {
                        int64_t ix = (o * dsize + k) * inner + j;
                        dot += static_cast<double>(g[ix]) * s[ix];
                    }
                    for (int64_t k = 0; k < dsize; ++k) {
                        int64_t ix = (o * dsize + k) * inner + j;
                        ag[ix] += s[ix] * (g[ix] - static_cast<float>(dot));
                    }
                }
            }
        };
    }
    return out;
}

Tensor log_softmax(const Tensor& a, int64_t dim) {
    dim = norm_dim(dim, a.ndim());
    Tensor out = make_out(a.shape(), a.requires_grad());
    int64_t outer = 1, inner = 1, dsize = a.dim(dim);
    for (int64_t d = 0; d < dim; ++d)
        outer *= a.dim(d);
    for (int64_t d = dim + 1; d < a.ndim(); ++d)
        inner *= a.dim(d);
    const float* ad = a.data();
    float* od = out.data();
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t j = 0; j < inner; ++j) {
            float m = -1e30f;
            for (int64_t k = 0; k < dsize; ++k) {
                float v = ad[(o * dsize + k) * inner + j];
                if (v > m)
                    m = v;
            }
            double s = 0;
            for (int64_t k = 0; k < dsize; ++k)
                s += std::exp(ad[(o * dsize + k) * inner + j] - m);
            float lse = m + static_cast<float>(std::log(s));
            for (int64_t k = 0; k < dsize; ++k)
                od[(o * dsize + k) * inner + j] = ad[(o * dsize + k) * inner + j] - lse;
        }
    }
    if (out.requires_grad()) {
        auto ao = a.impl(), oo = out.impl();
        oo->prev = {ao};
        oo->backward_fn = [ao, oo, dim]() {
            if (!ao->requires_grad)
                return;
            ensure_grad(ao);
            int64_t outer = 1, inner = 1, dsize = ao->shape[static_cast<size_t>(dim)];
            for (int64_t d = 0; d < dim; ++d)
                outer *= ao->shape[static_cast<size_t>(d)];
            for (int64_t d = dim + 1; d < static_cast<int64_t>(ao->shape.size()); ++d)
                inner *= ao->shape[static_cast<size_t>(d)];
            const float* y = oo->data.data(); // log-probs
            const float* g = oo->grad.data();
            float* ag = ao->grad.data();
            for (int64_t o = 0; o < outer; ++o) {
                for (int64_t j = 0; j < inner; ++j) {
                    double gsum = 0;
                    for (int64_t k = 0; k < dsize; ++k)
                        gsum += g[(o * dsize + k) * inner + j];
                    for (int64_t k = 0; k < dsize; ++k) {
                        int64_t ix = (o * dsize + k) * inner + j;
                        float s = std::exp(y[ix]); // softmax
                        ag[ix] += g[ix] - s * static_cast<float>(gsum);
                    }
                }
            }
        };
    }
    return out;
}

Tensor nll_loss(const Tensor& log_probs, const std::vector<int64_t>& targets) {
    if (log_probs.ndim() != 2)
        throw std::invalid_argument("nll_loss: log_probs must be 2D");
    int64_t N = log_probs.dim(0), C = log_probs.dim(1);
    if (static_cast<int64_t>(targets.size()) != N)
        throw std::invalid_argument("nll_loss: targets size mismatch");
    Tensor out = make_out({1}, log_probs.requires_grad());
    double acc = 0;
    for (int64_t i = 0; i < N; ++i) {
        int64_t t = targets[static_cast<size_t>(i)];
        if (t < 0 || t >= C)
            throw std::invalid_argument("nll_loss: target out of range");
        acc -= log_probs.data()[i * C + t];
    }
    out.data()[0] = static_cast<float>(acc / N);
    if (out.requires_grad()) {
        auto lo = log_probs.impl(), oo = out.impl();
        oo->prev = {lo};
        oo->backward_fn = [lo, oo, targets, N, C]() {
            if (!lo->requires_grad)
                return;
            ensure_grad(lo);
            float g = oo->grad[0] / static_cast<float>(N);
            for (int64_t i = 0; i < N; ++i)
                lo->grad[static_cast<size_t>(i * C + targets[static_cast<size_t>(i)])] -= g;
        };
    }
    return out;
}

Tensor cross_entropy(const Tensor& logits, const std::vector<int64_t>& targets) {
    return nll_loss(log_softmax(logits, -1), targets);
}

Tensor rmsnorm(const Tensor& a, const Tensor& weight, float eps) {
    // r = rsqrt(mean(a^2, -1, keepdim) + eps); out = a * r * weight
    Tensor a2 = mul(a, a);
    Tensor m = sum(a2, -1, true);
    int64_t last = a.dim(a.ndim() - 1); // original dim, not the keepdim 1
    Tensor ms = div(m, full(m.shape(), static_cast<float>(last)));
    Tensor mse = add(ms, full(ms.shape(), eps));
    Tensor r = pow(mse, -0.5f);
    Tensor ar = mul(a, r);
    return mul(ar, weight); // weight broadcasts over leading dims
}

Tensor dropout(const Tensor& a, float p, uint64_t seed) {
    if (p <= 0.0f)
        return a; // identity: graph flows through untouched
    if (p >= 1.0f)
        throw std::invalid_argument("dropout: p must be < 1");
    // Build mask with the same deterministic RNG as rand_uniform.
    Tensor mask = make_out(a.shape(), false);
    {
        // xorshift64* matching tensor.cpp's Rng
        uint64_t s = seed ? seed : 0x9E3779B97F4A7C15ULL;
        auto next = [&]() {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            return s * 0x2545F4914F6CDD1DULL;
        };
        for (int64_t i = 0; i < mask.numel(); ++i) {
            double u = (next() >> 11) * (1.0 / 9007199254740992.0);
            mask.data()[i] = (u >= p) ? 1.0f : 0.0f;
        }
    }
    float scale = 1.0f / (1.0f - p);
    // out = (a * mask) * scale, composed from primitives so autograd handles it.
    return mul(mul(a, mask), full(a.shape(), scale));
}

Tensor rope(const Tensor& x, const Tensor& cos, const Tensor& sin) {
    // x: [B,T,H,D] D even; cos/sin: [T,D/2]
    if (x.ndim() != 4 || cos.ndim() != 2 || sin.ndim() != 2)
        throw std::invalid_argument("rope: rank mismatch");
    int64_t B = x.dim(0), T = x.dim(1), H = x.dim(2), D = x.dim(3);
    if (D % 2 != 0)
        throw std::invalid_argument("rope: head dim must be even");
    int64_t Dh = D / 2;
    if (cos.dim(0) != T || cos.dim(1) != Dh || sin.dim(0) != T || sin.dim(1) != Dh)
        throw std::invalid_argument("rope: cos/sin shape mismatch");
    Tensor out = make_out(x.shape(), x.requires_grad());
    const float* xd = x.data();
    const float* cd = cos.data();
    const float* sd = sin.data();
    float* od = out.data();
    for (int64_t b = 0; b < B; ++b)
        for (int64_t t = 0; t < T; ++t)
            for (int64_t h = 0; h < H; ++h)
                for (int64_t d = 0; d < Dh; ++d) {
                    int64_t base = ((b * T + t) * H + h) * D;
                    float x1 = xd[base + d], x2 = xd[base + Dh + d];
                    float c = cd[t * Dh + d], s = sd[t * Dh + d];
                    od[base + d] = x1 * c - x2 * s;
                    od[base + Dh + d] = x1 * s + x2 * c;
                }
    if (out.requires_grad()) {
        auto xo = x.impl(), oo = out.impl();
        auto co = cos.impl(), so = sin.impl();
        oo->prev = {xo, co, so};
        oo->backward_fn = [xo, co, so, oo, B, T, H, Dh]() {
            const float* g = oo->grad.data();
            const float* cd = co->data.data();
            const float* sd = so->data.data();
            int64_t D = Dh * 2;
            if (xo->requires_grad) {
                ensure_grad(xo);
                float* xg = xo->grad.data();
                for (int64_t b = 0; b < B; ++b)
                    for (int64_t t = 0; t < T; ++t)
                        for (int64_t h = 0; h < H; ++h)
                            for (int64_t d = 0; d < Dh; ++d) {
                                int64_t base = ((b * T + t) * H + h) * D;
                                float g1 = g[base + d], g2 = g[base + Dh + d];
                                float c = cd[t * Dh + d], s = sd[t * Dh + d];
                                // rotation transpose: [c, s; -s, c]
                                xg[base + d] += g1 * c + g2 * s;
                                xg[base + Dh + d] += -g1 * s + g2 * c;
                            }
            }
            if (co->requires_grad || so->requires_grad) {
                // cos/sin are constants in practice; support grad anyway
                const float* xd = xo->data.data();
                if (co->requires_grad) {
                    ensure_grad(co);
                    float* cg = co->grad.data();
                    for (int64_t b = 0; b < B; ++b)
                        for (int64_t t = 0; t < T; ++t)
                            for (int64_t h = 0; h < H; ++h)
                                for (int64_t d = 0; d < Dh; ++d) {
                                    int64_t base = ((b * T + t) * H + h) * D;
                                    float x1 = xd[base + d], x2 = xd[base + Dh + d];
                                    float g1 = g[base + d], g2 = g[base + Dh + d];
                                    cg[t * Dh + d] += g1 * x1 + g2 * x2;
                                }
                }
                if (so->requires_grad) {
                    ensure_grad(so);
                    float* sg = so->grad.data();
                    for (int64_t b = 0; b < B; ++b)
                        for (int64_t t = 0; t < T; ++t)
                            for (int64_t h = 0; h < H; ++h)
                                for (int64_t d = 0; d < Dh; ++d) {
                                    int64_t base = ((b * T + t) * H + h) * D;
                                    float x1 = xd[base + d], x2 = xd[base + Dh + d];
                                    float g1 = g[base + d], g2 = g[base + Dh + d];
                                    sg[t * Dh + d] += -g1 * x2 + g2 * x1;
                                }
                }
            }
        };
    }
    return out;
}

} // namespace tt
