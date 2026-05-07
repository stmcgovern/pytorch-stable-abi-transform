#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/csrc/stable/device.h>
#include <cuda_runtime.h>

void foo(torch::stable::Tensor& out, const torch::stable::Tensor& input, double epsilon) {
    STD_TORCH_CHECK(input.dim() == 2, "expected 2D");
    float* ptr = input.mutable_data_ptr<float>();
    auto result = torch::stable::clone(input);
}

template<typename T>
void templated_fn(torch::stable::Tensor& t) {
    auto result = torch::stable::clone(t);
    T* ptr = t.mutable_data_ptr<T>();
}

void instantiate(torch::stable::Tensor& t) {
    templated_fn<float>(t);
    templated_fn<double>(t);
}
