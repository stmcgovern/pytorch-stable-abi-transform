#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

void kernel(torch::Tensor& out, const at::Tensor& input) {
    TORCH_CHECK(input.is_contiguous(), "must be contiguous");
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    float* ptr = input.data_ptr<float>();
    auto result = input.clone();
}
