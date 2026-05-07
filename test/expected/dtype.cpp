#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/csrc/stable/device.h>

void check_types(torch::stable::Tensor& a, const torch::stable::Tensor& b) {
    auto dt = a.scalar_type();
    bool same = (a.scalar_type() == b.scalar_type());
}
