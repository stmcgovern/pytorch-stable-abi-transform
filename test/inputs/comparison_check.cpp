#include <torch/all.h>

int compute(int a, int b);

void validate(const at::Tensor& t, int n) {
    TORCH_CHECK_EQ(t.dim(), 2);
    TORCH_CHECK_NE(t.size(0), 0);
    TORCH_CHECK_LT(n, t.size(1));
    TORCH_CHECK_GT(t.numel(), 0);
    TORCH_CHECK_GE(t.dim(), 1);
    TORCH_CHECK_LE(n, 100);
}

// Precedence: without parenthesization, a + b == c * d
// would be parsed as a + (b == (c * d)) by C++
void check_precedence(int a, int b, int c, int d) {
    TORCH_CHECK_EQ(a + b, c * d);
    TORCH_CHECK_LT(a | b, c & d);
}

// Nested function call with comma — verifies MacroArgs
// correctly identifies compute(a, b) as a single argument
void check_nested(int a, int b, int c) {
    TORCH_CHECK_GE(compute(a, b), c);
}
