#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/csrc/stable/device.h>

void methods(torch::stable::Tensor& t) {
    auto a = torch::stable::contiguous(t);
    auto b = torch::stable::clone(t);
    auto c = torch::stable::view(t, {-1});
    auto d = torch::stable::reshape(t, {2, 3});
    torch::stable::zero_(t);
    auto e = torch::stable::transpose(t, 0, 1);
}
