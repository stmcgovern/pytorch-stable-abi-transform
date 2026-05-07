#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/csrc/stable/device.h>
#include <cuda_runtime.h>

void launch(torch::stable::Tensor& input) {
    const torch::stable::accelerator::DeviceGuard device_guard(input.get_device_index());
    void* stream_ptr = nullptr;
    aoti_torch_get_current_cuda_stream(input.get_device_index(), &stream_ptr);
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
}
