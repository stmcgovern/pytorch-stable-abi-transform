#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/csrc/stable/device.h>

void kernel(torch::stable::Tensor& out, const torch::stable::Tensor& input) {
    float* ptr = input.mutable_data_ptr<float>();
    auto result = torch::stable::clone(input);
}
