#include <torch/all.h>

void validate(const at::Tensor& t, int n) {
    TORCH_CHECK_EQ(t.dim(), 2);
    TORCH_CHECK_NE(t.size(0), 0);
    TORCH_CHECK_LT(n, t.size(1));
    TORCH_CHECK_GT(t.numel(), 0);
    TORCH_CHECK_GE(t.dim(), 1);
    TORCH_CHECK_LE(n, 100);
}
