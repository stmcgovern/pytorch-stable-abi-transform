#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

void launch(torch::Tensor& input) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
}
