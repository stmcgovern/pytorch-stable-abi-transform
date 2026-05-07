#include <torch/all.h>

void kernel(torch::Tensor& out, const at::Tensor& input) {
    float* ptr = input.data_ptr<float>();
    auto result = input.clone();
}
