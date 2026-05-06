# PyTorch Stable ABI Transform

Clang-based source-to-source rewriter that migrates PyTorch C++ extensions from the unstable ABI (`at::`, `c10::`, `torch::`) to the [stable ABI](https://github.com/pytorch/pytorch/wiki/Stable-ABI) (`torch::stable::`, `torch::headeronly::`).

The stable ABI lets extensions work across PyTorch versions without recompilation. This tool handles the mechanical migration — includes, types, macros, `data_ptr` disambiguation, CUDA guards, method-to-function rewrites — so you focus on the ~1% that needs judgment.

## Quick start

```bash
# Build (requires LLVM/Clang 19 dev libs)
mkdir build && cd build && cmake .. && make -j$(nproc)

# Generate config
./build/stable-abi-transform --init-config > .stable-abi.yaml
# Edit: set pytorch_root, project_root, include_paths

# Audit (read-only)
./build/stable-abi-transform                        # mode: audit

# Preview changes
# set mode: rewrite in .stable-abi.yaml
./build/stable-abi-transform --dry-run

# Rewrite + auto-verify
./build/stable-abi-transform
```

Before/after — the tool rewrites this:

```cpp
#include <torch/all.h>
void foo(const at::Tensor& input) {
    TORCH_CHECK(input.dim() == 2);
    float* ptr = input.data_ptr<float>();
    auto result = input.clone();
}
```

To this:

```cpp
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
void foo(const torch::stable::Tensor& input) {
    STD_TORCH_CHECK(input.dim() == 2);
    const float* ptr = input.const_data_ptr<float>();
    auto result = torch::stable::clone(input);
}
```

Patterns it can't auto-rewrite (TensorOptions decomposition, PYBIND11_MODULE replacement, project dispatch macros) are flagged with actionable guidance.

## Verification

Compile-based verification creates a shadow include tree with *only* stable headers. If the file compiles against it, migration is provably complete — zero false positives, zero false negatives. See [docs/user-guide.md](docs/user-guide.md) for details.

## Documentation

- **[User Guide](docs/user-guide.md)** — full reference: modes, config, flags, verification, architecture, rule regeneration, CI integration
- **[Claude Skill](docs/claude-skill.md)** — Claude Code skill for agent-assisted migration
- **[CLAUDE.md](CLAUDE.md)** — project context for Claude Code

## License

[MIT](LICENSE)
