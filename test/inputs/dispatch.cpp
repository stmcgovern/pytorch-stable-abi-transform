#include <torch/all.h>
#include <ATen/Dispatch.h>

void dispatch_kernel(const torch::Tensor& input) {
    AT_DISPATCH_SWITCH(input.scalar_type(), "dispatch_test",
        AT_DISPATCH_CASE(at::ScalarType::Float, [&] {
            auto* ptr = input.data_ptr<scalar_t>();
        })
        AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
            auto* ptr = input.data_ptr<scalar_t>();
        })
    );
}
