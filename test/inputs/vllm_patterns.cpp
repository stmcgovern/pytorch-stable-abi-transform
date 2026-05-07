#include <torch/all.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

// Pattern 1: Switch on ScalarType with void data_ptr + reinterpret_cast
// (from vLLM custom_all_reduce.cu)
void manual_type_dispatch(torch::Tensor& out, void* buffer, int64_t n) {
    switch (out.scalar_type()) {
    case at::ScalarType::Float: {
        auto* ptr = reinterpret_cast<float*>(out.data_ptr());
        break;
    }
    case at::ScalarType::Half: {
        auto* ptr = reinterpret_cast<at::Half*>(out.data_ptr());
        break;
    }
    case at::ScalarType::BFloat16: {
        auto* ptr = reinterpret_cast<at::BFloat16*>(out.data_ptr());
        break;
    }
    default:
        break;
    }
}

// Pattern 2: Template kernel launcher with const/mutable data_ptr
// and multiple sizes/strides calls (from vLLM layernorm, attention kernels)
template<typename scalar_t>
void launch_kernel(const at::Tensor& input, at::Tensor& output) {
    const scalar_t* in_ptr = input.data_ptr<scalar_t>();
    scalar_t* out_ptr = output.data_ptr<scalar_t>();
    auto dim0 = input.sizes()[0];
    auto dim1 = input.sizes()[1];
    auto stride0 = input.strides()[0];
}

// Pattern 3: AT_DISPATCH calling template launcher + elementSize
// (common vLLM dispatch-to-template pattern)
void dispatch_to_kernel(const torch::Tensor& input, torch::Tensor& output) {
    auto nbytes = input.numel() * c10::elementSize(input.scalar_type());
    AT_DISPATCH_SWITCH(input.scalar_type(), "kernel",
        AT_DISPATCH_CASE(at::ScalarType::Float, [&] {
            launch_kernel<scalar_t>(input, output);
        })
        AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
            launch_kernel<scalar_t>(input, output);
        })
    );
}

// Pattern 4: Free functions — at::empty
// (from vLLM workspace allocation patterns)
torch::Tensor alloc_workspace(at::Tensor& src) {
    auto workspace = at::empty({src.size(0), 256}, src.scalar_type());
    return workspace;
}

// Pattern 5: Multiple data_ptr on same line, const detection
// (from vLLM fused kernel launches)
void multi_ptr(at::Tensor& a, const at::Tensor& b) {
    float* a_ptr = a.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
}

// Pattern 6: CUDA stream + DeviceGuard combo
// (from nearly every vLLM kernel launcher)
void guarded_launch(const torch::Tensor& input, torch::Tensor& output) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    auto dt = input.dtype();
    auto result = input.clone();
}

// Pattern 7: data_ptr passed directly as function arguments (const vs mutable)
// (from vLLM CUDA kernel launches — the callee parameter type determines constness)
void kernel_func(float* out, const float* in, int n);
void direct_arg_pattern(torch::Tensor& out, const torch::Tensor& input) {
    kernel_func(out.data_ptr<float>(), input.data_ptr<float>(), 42);
}

// Pattern 8: data_ptr through std::optional operator->
// (from vLLM pos_encoding_kernels.cu — optional key tensor)
void optional_pattern(std::optional<torch::Tensor>& opt) {
    if (opt.has_value()) {
        float* p = opt->data_ptr<float>();
    }
}
