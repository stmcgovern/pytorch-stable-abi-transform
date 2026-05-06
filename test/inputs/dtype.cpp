#include <torch/all.h>

void check_types(torch::Tensor& a, const torch::Tensor& b) {
    auto dt = a.dtype();
    bool same = (a.dtype() == b.dtype());
}
