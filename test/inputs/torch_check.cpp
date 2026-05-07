#include <torch/all.h>

void check(const torch::Tensor& t) {
    TORCH_CHECK(t.is_contiguous(), "must be contiguous");
    TORCH_CHECK(t.dim() == 2);
}
