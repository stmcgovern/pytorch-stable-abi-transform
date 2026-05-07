#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/csrc/stable/device.h>

void check(const torch::stable::Tensor& t) {
    STD_TORCH_CHECK(t.is_contiguous(), "must be contiguous");
    STD_TORCH_CHECK(t.dim() == 2);
}
