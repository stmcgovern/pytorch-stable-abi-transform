# stable-abi-transform

Clang LibTooling source-to-source rewriter that transforms PyTorch C++ extension code from the unstable internal ABI to the stable ABI. Generic ecosystem tool — works on any PyTorch C++ consumer.

## Build

```bash
dnf install -y clang-devel llvm-devel  # Fedora 41, LLVM 19
mkdir build && cd build && cmake .. && make -j$(nproc)
```

Tested with: LLVM 19, CUDA 12.9, C++20.

## Test

```bash
PYTORCH_DIR=/path/to/pytorch RESOURCE_DIR=/usr/lib/clang/19 bash test/run_tests.sh
```

11 test cases: input/expected pairs in `test/inputs/` and `test/expected/`.

## Usage

See [README.md](README.md) for quick start and [docs/user-guide.md](docs/user-guide.md) for full reference.

The tool is config-driven via `.stable-abi.yaml` — run `stable-abi-transform --init-config` to generate one.

## Architecture

- `src/main.cpp` — CLI, mode dispatch, ClangTool configuration
- `src/StableAbiAction.{h,cpp}` — FrontendAction + ASTConsumer wiring
- `src/TransformerRules.cpp` — AST-based rewrite rules (Clang Transformer)
- `src/AstCallbacks.{h,cpp}` — Manual AST matchers (CUDA guard/stream)
- `src/PreprocessorCallbacks.cpp` — Include and macro rewrites
- `src/Reporter.{h,cpp}` — Finding accumulation, text/JSON output
- `src/Verifier.{h,cpp}` — Compile-based and regex-based verification
- `src/Rules.h` — Auto-generated transformation tables (from `scripts/gen_rules.py`)
- `src/Config.{h,cpp}` — YAML config support

## Key design decisions

- Compile-based verification uses a shadow include tree with only stable headers. Sound because stable headers form a closed dependency set — they never include `c10/`, `ATen/`, or `torch/csrc/api/`.
- `data_ptr<T>()` disambiguation uses AST const-qualification analysis to choose `mutable_data_ptr` vs `const_data_ptr`.
- Rules.h is auto-generated from PyTorch headers by `scripts/gen_rules.py`. Include rules are hand-maintained (stable headers don't encode which unstable headers map to them).
- CUDA stream rewriting uses `aoti_torch_get_current_cuda_stream()` (C shim) rather than the C++ Stream class, because the C++ class doesn't expose raw `cudaStream_t`.
