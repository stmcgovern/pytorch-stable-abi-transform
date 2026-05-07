#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

void foo(torch::Tensor& out, const at::Tensor& input, double epsilon) {
    TORCH_CHECK(input.dim() == 2, "expected 2D");
    float* ptr = input.data_ptr<float>();
    auto result = input.clone();
}

template<typename T>
void templated_fn(at::Tensor& t) {
    auto result = t.clone();
    T* ptr = t.data_ptr<T>();
}

void instantiate(at::Tensor& t) {
    templated_fn<float>(t);
    templated_fn<double>(t);
}
