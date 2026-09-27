#include <gtest/gtest.h>

#include "tinytrain/tensor.h"

TEST(Scaffold, TensorShapeAndFill) {
    tt::Tensor t({2, 3}, 1.5f);
    EXPECT_EQ(t.ndim(), 2);
    EXPECT_EQ(t.numel(), 6);
    EXPECT_EQ(t.dim(0), 2);
    EXPECT_EQ(t.dim(1), 3);
    for (int64_t i = 0; i < t.numel(); ++i)
        EXPECT_FLOAT_EQ(t.data()[i], 1.5f);
    EXPECT_EQ(t.offset({1, 2}), 5);
}

TEST(Scaffold, TensorRejectsBadShape) {
    EXPECT_THROW(tt::Tensor(std::vector<int64_t>{}), std::invalid_argument);
    EXPECT_THROW(tt::Tensor({2, 0}), std::invalid_argument);
    EXPECT_THROW(tt::Tensor({2, -1}), std::invalid_argument);
}
