#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/csrc/stable/device.h>

void validate(const torch::stable::Tensor& t, int n) {
    STD_TORCH_CHECK((t.dim()) == (2));
    STD_TORCH_CHECK((t.size(0)) != (0));
    STD_TORCH_CHECK((n) < (t.size(1)));
    STD_TORCH_CHECK((t.numel()) > (0));
    STD_TORCH_CHECK((t.dim()) >= (1));
    STD_TORCH_CHECK((n) <= (100));
}
