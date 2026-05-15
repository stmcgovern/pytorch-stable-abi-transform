#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/csrc/stable/device.h>

void process(std::optional<torch::stable::Tensor> t, std::optional<int> scale) {
    if (t.has_value()) {
        STD_TORCH_CHECK(scale.has_value(), "need scale");
    }
}

std::optional<int> maybe_get() {
    return std::nullopt;
}

::std::optional<int> maybe_get2() {
    return std::nullopt;
}
