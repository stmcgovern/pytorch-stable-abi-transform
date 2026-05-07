#include <torch/all.h>

void methods(torch::Tensor& t) {
    auto a = t.contiguous();
    auto b = t.clone();
    auto c = t.view({-1});
    auto d = t.reshape({2, 3});
    t.zero_();
    auto e = t.transpose(0, 1);
}
