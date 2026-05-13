#include <torch/all.h>

void process(c10::optional<at::Tensor> t, c10::optional<int> scale) {
    if (t.has_value()) {
        TORCH_CHECK(scale.has_value(), "need scale");
    }
}

c10::optional<int> maybe_get() {
    return c10::nullopt;
}

::c10::optional<int> maybe_get2() {
    return ::c10::nullopt;
}
