---
name: migrate-stable-abi
description: Migrate PyTorch C++ extensions from unstable ABI to stable ABI using the stable-abi-transform tool. Use when the user mentions stable ABI, migrating CUDA kernels or C++ extensions, auditing unstable API usage, or verifying stable ABI compliance.
---

You migrate PyTorch C++ extensions to the stable ABI — a guaranteed-stable API surface that lets extensions work across PyTorch versions without recompilation.

Your primary tool is `stable-abi-transform`, a Clang-based AST rewriter that mechanically transforms ~99% of unstable API patterns (type renames, include swaps, const/mutable data_ptr disambiguation, macro rewrites, DeviceGuard conversion). You provide judgment for the ~1% the tool flags: TensorOptions decomposition, op registration migration, project-specific dispatch macros. A compile-based verifier provides a sound "done" signal — if the file compiles against only stable headers, migration is provably complete.

Don't rewrite PyTorch API calls by hand — the tool's AST analysis handles subtle patterns (const-qualification chains, preprocessor macros, namespace catch-alls) that text manipulation would miss. Locate the tool on PATH or build from source (`mkdir build && cd build && cmake .. && make -j$(nproc)`).

## 1. Completeness Invariant

A file is fully migrated when `--mode=verify --verify-method=compile` returns exit code 0. This compiles the file against a shadow include tree containing *only* stable PyTorch headers. Stable headers form a closed dependency set (they never include `c10/`, `ATen/`, or `torch/csrc/api/`), so this check has zero false positives and zero false negatives.

The workflow is self-correcting: the tool transforms what it can, you handle flags, and the compile verifier catches anything either missed. Additionally, the tool has AST-based catch-all matchers that flag ANY remaining `at::`, `c10::`, or unstable `torch::` usage — no unstable pattern can silently pass through unreported.

## 2. Setup

Before migrating, configure the tool for the target project.

### Locate PyTorch root

The source tree or install prefix containing stable headers. Verify:
```bash
ls $PYTORCH/torch/csrc/stable/tensor.h  # must exist
```

### Create config

```bash
stable-abi-transform --init-config > .stable-abi.yaml
```

Edit the generated file. Example:
```yaml
mode: audit                               # switch to rewrite or verify as you go
pytorch_root: /path/to/pytorch
project_root: ./csrc
verify_method: compile                    # compile (sound) or regex (fallback)

compiler_flags:
  - -std=c++20

include_paths:
  - ${pytorch_root}/torch/csrc/api/include
  - ${pytorch_root}
  - ${pytorch_root}/torch/include
  - /usr/local/cuda/include
  - ./csrc/inc                            # project-specific headers

extra_includes:                           # for verify mode — project header dirs
  - ./csrc
```

At minimum you need: `pytorch_root`, the three PyTorch include paths, CUDA include path, and the project's own header directories. All flags live in the config — the only command is `stable-abi-transform`.

### Test the config

```bash
stable-abi-transform  # auto-loads .stable-abi.yaml, runs audit
```

If you see parse errors, fix missing include paths or compiler flags before proceeding.

## 3. The Migration Loop

### Phase 1 — Mechanical transform

Set `mode: rewrite` in `.stable-abi.yaml` and run:

```bash
stable-abi-transform                    # rewrites all files under project_root
stable-abi-transform --dry-run          # preview as unified diff first
```

The tool handles all mechanical patterns: include paths, type renames, scalar/device shorthands, macro renames, `TORCH_LIBRARY` → `STABLE_TORCH_LIBRARY`, `data_ptr<T>()` → `mutable/const_data_ptr<T>()` (using AST const-analysis), DeviceGuard/stream conversion, and method-to-free-function rewrites. Don't duplicate this work by hand.

After rewrite, the tool automatically runs post-rewrite verification. If exit 0, the file is done. Note: post-rewrite verification uses compile-based checking only if `pytorch_root` is set (in config or via `--pytorch-root`); otherwise it falls back to weaker regex checking. Always set `pytorch_root` for reliable results.

### Phase 2 — Handle flags

If the tool reports `[FLAG]` findings, handle each manually:

| Flag Pattern | Action |
|---|---|
| `TensorOptions(...)` | Decompose into individual params: `torch::stable::empty(shape, dtype, layout, device, pin_memory)` |
| `PYBIND11_MODULE` | Replace with `STABLE_TORCH_LIBRARY_FRAGMENT` + `STABLE_TORCH_LIBRARY_IMPL` + `TORCH_BOX()` (see Section 4d) |
| `elementSize(dtype_var)` | Rewrite to use a tensor method: `tensor.element_size()`, or compute manually from ScalarType |
| Project dispatch macros (e.g. `VLLM_DISPATCH_*`) | Create stable versions using `THO_DISPATCH_SWITCH`/`THO_DISPATCH_CASE` (see Section 4b) |
| DeviceGuard/CudaStream in macro body | Extract guard+stream code out of the macro, or rewrite the macro definition |
| `getCurrentCUDAStream` (non-standard usage) | Use `aoti_torch_get_current_cuda_stream()` C shim, or wrap in project helper (see Section 4a) |

### Phase 3 — Verify and fix

Set `mode: verify` and `verify_method: compile` in `.stable-abi.yaml`, then:

```bash
stable-abi-transform
```

Ensure `extra_includes` lists the project's own header directories (e.g., `./csrc`) — the verifier needs them to resolve `#include "project_header.h"`.

Exit 0 = done. Exit 1 = read the compiler errors and fix. Common failures:

| Compile Error | Fix |
|---|---|
| Missing include (unknown type) | Add the appropriate `<torch/csrc/stable/...>` or `<torch/headeronly/...>` header |
| `device_of()` undefined | Replace with `tensor.get_device_index()` |
| `c10::Scalar` in function signature | Refactor parameter to `double` or `int64_t` — TORCH_BOX cannot box `c10::Scalar` |
| `at::acc_type<T>` undefined | Hardcode the accumulation type (typically `float` for half/bfloat16 inputs) |
| `torch::autograd` undefined | Flag to user — requires architectural change, not mechanical rewrite. Autograd has no stable ABI equivalent. |

Repeat: fix → verify → fix → verify, until exit 0. If a cycle doesn't reduce the number of compile errors, stop and report the remaining errors to the user — they likely require architectural decisions (e.g., autograd removal, API redesign) that the agent shouldn't make unilaterally.

## 4. Project Infrastructure

For multi-file projects, create infrastructure files in dependency order: `torch_utils.h` and `dispatch_utils.h` first (they're independent), then migrate kernel files, then create `ops.h` and `torch_bindings.cpp` last (they depend on the migrated kernel signatures).

### 4a. `torch_utils.h` — Convenience header

Bundles stable includes and provides a CUDA stream helper. Template (from vLLM):

```cpp
#pragma once

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/shim_utils.h>

#include <cuda_runtime.h>

#define STD_TORCH_CHECK_NOT_IMPLEMENTED(cond, ...) \
  STD_TORCH_CHECK(cond, "NotImplementedError: ", __VA_ARGS__)

inline cudaStream_t get_current_cuda_stream(int32_t device_index = -1) {
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(device_index, &stream_ptr));
  return reinterpret_cast<cudaStream_t>(stream_ptr);
}
```

### 4b. `dispatch_utils.h` — Stable dispatch macros

Adapt the project's existing `AT_DISPATCH_*` macros. Mechanical translation:
- `AT_DISPATCH_SWITCH` → `THO_DISPATCH_SWITCH`
- `AT_DISPATCH_CASE` → `THO_DISPATCH_CASE`
- `at::ScalarType::X` → `torch::headeronly::ScalarType::X`

Template (from vLLM):

```cpp
#pragma once

#include <torch/headeronly/core/Dispatch.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#define PROJECT_STABLE_DISPATCH_CASE_FLOATING_TYPES(...)                  \
  THO_DISPATCH_CASE(torch::headeronly::ScalarType::Float, __VA_ARGS__)   \
  THO_DISPATCH_CASE(torch::headeronly::ScalarType::Half, __VA_ARGS__)    \
  THO_DISPATCH_CASE(torch::headeronly::ScalarType::BFloat16, __VA_ARGS__)

#define PROJECT_STABLE_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  THO_DISPATCH_SWITCH(TYPE, NAME,                               \
                      PROJECT_STABLE_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))
```

Read the project's existing dispatch_utils.h and create a stable equivalent for each macro.

### 4c. `ops.h` — Op forward declarations

Forward-declare all migrated kernel entry points using stable types:

```cpp
#pragma once
#include <torch/csrc/stable/tensor.h>
#include <optional>

void my_kernel(torch::stable::Tensor& output,
               const torch::stable::Tensor& input,
               int64_t param);
```

### 4d. `torch_bindings.cpp` — Op registration

This is the most complex manual step. The key insight: `PYBIND11_MODULE` creates Python-callable C functions. `STABLE_TORCH_LIBRARY` registers ops in PyTorch's dispatcher. With the dispatcher, Python calls ops via `torch.ops.NAMESPACE.op_name()` — no pybind11 needed.

```cpp
#include <torch/csrc/stable/library.h>
#include "ops.h"

// Schema definitions — use PyTorch's operator schema syntax:
//   Tensor! = mutable tensor (in-place), Tensor = const tensor
//   int = int64_t, float = double, bool = bool
//   * = keyword-only separator, ? suffix = optional
STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
    ops.def("my_kernel(Tensor! output, Tensor input, int param) -> ()");
    ops.def("my_func(Tensor input, float scale) -> Tensor");
}

// Implementation registration with dispatch key
STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
    ops.impl("my_kernel", TORCH_BOX(&my_kernel));
    ops.impl("my_func", TORCH_BOX(&my_func));
}

// Composite ops (no device dispatch) use CompositeExplicitAutograd
STABLE_TORCH_LIBRARY_IMPL(_C, CompositeExplicitAutograd, ops) {
    ops.impl("cpu_only_op", TORCH_BOX(&cpu_only_op));
}
```

Use `STABLE_TORCH_LIBRARY_FRAGMENT` (not `STABLE_TORCH_LIBRARY`) when adding ops to a namespace — FRAGMENT allows multiple registration blocks, while non-FRAGMENT requires exactly one per namespace.

**Python module registration** is separate from op registration. The project needs its own mechanism (e.g., a minimal `PyMODINIT_FUNC` + `PyModule_Create`, or setuptools entry points). This is NOT provided by PyTorch's stable API. vLLM uses a helper macro `REGISTER_EXTENSION` in `csrc/core/registration.h` for this.

**Partial migration**: If `PYBIND11_MODULE` registers both tensor ops AND non-tensor utilities (config functions, version queries, etc.), only the tensor ops move to `STABLE_TORCH_LIBRARY`. Non-tensor functions can stay with pybind11 or move to a plain Python C extension. Python calls migrated ops via `torch.ops.NAMESPACE.op_name()`.

**TORCH_BOX supported parameter types**: `Tensor`, `int64_t`, `double`, `bool`, `std::string`, `ScalarType`, `DeviceType`, `Layout`, `MemoryFormat`, `Device`, `std::optional<T>`, `std::vector<T>` where T is any supported type. Functions taking `c10::Scalar` or custom types **cannot** be boxed — refactor parameters to primitive types.

## 5. Stable API Reference

When handling flags or compile errors, consult this to determine what's available.

### What does NOT have a stable equivalent

Check this first — if the source uses any of these, you cannot mechanically migrate it:

- **`packed_accessor32<T,N>()` / `packed_accessor64<T,N>()`**: Replace with raw pointers + explicit stride params: `tensor.mutable_data_ptr<T>()`, `tensor.size(d)`, `tensor.stride(d)`
- **`torch::autograd`**: Gradient tracking, backward, autograd tape. Requires architectural change.
- **`c10::Scalar` as function parameter**: TORCH_BOX can't box it. Refactor to `double` or `int64_t`.
- **`at::acc_type<T>`**: Hardcode the accumulation type (`float` for half/bfloat16).
- **`torch::nn` modules**: No stable equivalent. Custom ops must use raw tensor operations.

### What IS available

**Free functions** (`torch::stable::`): `empty`, `empty_like`, `full`, `from_blob`, `new_empty`, `new_zeros`, `clone`, `copy_`, `contiguous`, `view`, `reshape`, `transpose`, `flatten`, `squeeze`, `unsqueeze`, `select`, `narrow`, `pad`, `matmul`, `sum`, `sum_out`, `subtract`, `amax`, `zero_`, `fill_`, `to`, `parallel_for`, `get_num_threads`

**Tensor methods**: `dim`, `numel`, `size(d)`, `stride(d)`, `sizes`, `strides`, `scalar_type`, `device`, `is_contiguous`, `element_size`, `mutable_data_ptr<T>`, `const_data_ptr<T>`, `data_ptr`, `get_device_index`, `is_cuda`, `is_cpu`, `defined`, `storage_offset`, `layout`, `set_requires_grad`

**Types** (`torch::headeronly::`): `ScalarType`, `DeviceType`, `Layout`, `MemoryFormat`, `Half`, `BFloat16`, `Float8_e4m3fn`, `Float8_e5m2`, `IntHeaderOnlyArrayRef`

**Macros**: `STD_TORCH_CHECK`, `STD_CUDA_CHECK`, `STD_CUDA_KERNEL_LAUNCH_CHECK`

**Dispatch**: `THO_DISPATCH_SWITCH(type, name, ...)`, `THO_DISPATCH_CASE(enum_type, ...)`

**Device/stream**: `DeviceGuard(device_index)`, `getCurrentStream(device_index)`, `getCurrentDeviceIndex()` (all in `torch::stable::accelerator::`)

**Registration**: `STABLE_TORCH_LIBRARY`, `STABLE_TORCH_LIBRARY_FRAGMENT`, `STABLE_TORCH_LIBRARY_IMPL`, `TORCH_BOX`

### Version requirements

- **2.9+**: DeviceGuard, Stream, new_empty, new_zeros
- **2.10+**: empty, full, from_blob, view, reshape, contiguous, to, sum, subtract, parallel_for, STD_CUDA_CHECK, mutable/const_data_ptr
- **2.11+**: from_blob with custom deleter

## 6. Migration Order

For large projects:
1. **Utility headers** (torch_utils.h, dispatch_utils.h) — no dependencies, enable everything else
2. **Leaf `.cu` files** — kernel files that don't include other project kernel headers
3. **Shared `.cuh` headers** — included by multiple kernels
4. **`ops.h` + `torch_bindings.cpp`** — depend on all kernel signatures being finalized

Build the include graph: `grep -rn '#include "' csrc/ | grep -v '/build/'`. Migrate bottom-up.

### In-place vs copy-then-modify

Two migration strategies (project-level decision, ask the user):
- **In-place**: `--mode=rewrite` modifies files directly. Simpler. Use when the project can break backward compatibility.
- **Copy-then-modify**: Copy files to a new directory (e.g., `csrc/libtorch_stable/`), rewrite there. vLLM uses this to maintain parallel stable/unstable builds during transition.

## 7. Validation

A migration chunk is complete when all four hold:
1. `mode: verify` with `verify_method: compile` → `stable-abi-transform` returns exit 0
2. `mode: audit` → `stable-abi-transform` shows 0 findings
3. The project builds successfully
4. The project's tests pass (ask the user which tests are relevant)

## Your Job

### Assess

Set `mode: audit` in `.stable-abi.yaml` and run `stable-abi-transform`. Check: how many files? How many findings? How many flags? Any parse errors? This determines scope and approach.

If audit reports parse errors: fix them before migrating. Common causes:
- Missing CUDA device headers → add `-I/usr/local/cuda/include` (or your CUDA path) to config
- Missing project-internal headers → add the project's `csrc/` or `inc/` to `include_paths`
- Undefined macros from project config → add the required `-D` defines to `compiler_flags`
- Files including unreachable headers → check `project_root` and `include_paths` settings

### Decide

- **Few files, 0 flags** → straightforward. Run rewrite, verify, done.
- **Many files, flags present** → create infrastructure first (torch_utils.h, dispatch_utils.h), then migrate in dependency order (Section 6).
- **`PYBIND11_MODULE` flagged** → plan the op registration migration (Section 4d) early. Note: only tensor-related operations move to `STABLE_TORCH_LIBRARY`. Non-tensor utility functions (config, versioning, etc.) can stay with pybind11 or use a plain Python C extension.
- **`torch::autograd` detected** → flag to user immediately. This requires architectural decisions, not mechanical migration.

### Execute

Phase 1 → Phase 2 → Phase 3, per Section 3.

### Report

What was migrated, what flags needed manual handling, what compile errors remain (if any), and what needs user decisions (autograd, API redesign, out-of-scope patterns).
