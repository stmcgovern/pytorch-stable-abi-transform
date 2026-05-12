#include <torch/all.h>

void process(c10::optional<at::Tensor> t, c10::optional<int> scale) {
    if (t.has_value()) {
        TORCH_CHECK(scale.has_value(), "need scale");
    }
}
