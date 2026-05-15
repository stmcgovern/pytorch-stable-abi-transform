#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/csrc/stable/device.h>

int compute(int a, int b);

void validate(const torch::stable::Tensor& t, int n) {
    STD_TORCH_CHECK((t.dim()) == (2));
    STD_TORCH_CHECK((t.size(0)) != (0));
    STD_TORCH_CHECK((n) < (t.size(1)));
    STD_TORCH_CHECK((t.numel()) > (0));
    STD_TORCH_CHECK((t.dim()) >= (1));
    STD_TORCH_CHECK((n) <= (100));
}

// Precedence: without parenthesization, a + b == c * d
// would be parsed as a + (b == (c * d)) by C++
void check_precedence(int a, int b, int c, int d) {
    STD_TORCH_CHECK((a + b) == (c * d));
    STD_TORCH_CHECK((a | b) < (c & d));
}

// Nested function call with comma — verifies MacroArgs
// correctly identifies compute(a, b) as a single argument
void check_nested(int a, int b, int c) {
    STD_TORCH_CHECK((compute(a, b)) >= (c));
}
