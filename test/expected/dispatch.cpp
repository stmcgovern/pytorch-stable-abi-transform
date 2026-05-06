#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/csrc/stable/device.h>
#include <torch/headeronly/core/Dispatch.h>
#include <torch/headeronly/core/Dispatch_v2.h>

void dispatch_kernel(const torch::stable::Tensor& input) {
    THO_DISPATCH_SWITCH(input.scalar_type(), "dispatch_test",
        THO_DISPATCH_CASE(torch::headeronly::ScalarType::Float, [&] {
            auto* ptr = input.mutable_data_ptr<scalar_t>();
        })
        THO_DISPATCH_CASE(torch::headeronly::ScalarType::Half, [&] {
            auto* ptr = input.mutable_data_ptr<scalar_t>();
        })
    );
}
