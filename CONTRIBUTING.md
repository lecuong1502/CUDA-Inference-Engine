# Contributing to CUDA Inference Engine

First off — thanks for taking the time to look at this project. This is a personal learning project focused on GPU inference engineering, and contributions that deepen that exploration are very welcome.

---

## What kind of contributions are welcome?

**Most welcome:**
- Bug fixes in CUDA kernels (incorrect output, numerical instability, race conditions)
- Performance improvements with measured benchmarks to back them up
- New kernel variants (e.g. FP16 attention, INT4 quantization, grouped-query attention)
- Better test coverage for numerical correctness
- Documentation fixes or clearer explanations in code comments

**Also welcome:**
- New model support beyond GPT-2 (LLaMA-style RoPE, SwiGLU FFN, etc.)
- Build system improvements (CMake, CI via GitHub Actions)
- Python API ergonomics

**Out of scope for now:**
- Training support — this engine is inference-only by design
- Replacing hand-written kernels with cuBLAS/cuDNN — the whole point is to understand what's happening inside

---

## Getting started

### Prerequisites

```bash
# CUDA 12.x
nvcc --version

# CMake 3.20+
cmake --version

# Python 3.10+
pip install pybind11 safetensors numpy torch  # torch only needed for correctness tests
```

### Build from source

```bash
git clone https://github.com/lecuong1502/CUDA-Inference-Engine
cd CUDA-Inference-Engine

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Run tests

```bash
# Numerical correctness vs PyTorch reference
python tests/test_kernels.py

# Full benchmark suite
python benchmarks/run_benchmark.py
```

---

## Workflow

1. **Open an issue first** for anything non-trivial — kernel changes, new features, API modifications. Describe what you're trying to fix or add and why.

2. **Fork and branch** from `main`:
   ```bash
   git checkout -b fix/attention-kernel-overflow
   ```
   Branch naming: `fix/`, `feat/`, `bench/`, `docs/` prefixes.

3. **Write or update tests.** Kernel changes must include a test in `tests/test_kernels.py` that validates output against a PyTorch reference within an acceptable tolerance:
   ```python
   assert torch.allclose(engine_output, pytorch_output, atol=1e-4, rtol=1e-3)
   ```

4. **Include benchmark numbers** for any change that claims a performance improvement. Format:
   ```
   Before: X ms/token, Y tok/s (RTX 4050, seq_len=512)
   After:  X ms/token, Y tok/s
   ```

5. **Open a pull request** against `main`. Keep the PR focused — one fix or feature per PR.

---

## Coding style

### CUDA / C++

- Follow the existing file structure: one kernel family per `.cu` file (`gemm.cu`, `attention.cu`, etc.)
- Use `// ===` section headers to separate kernel variants within a file
- Document launch parameters at the top of each kernel:
  ```cpp
  // Grid:  (M/BM, N/BN)
  // Block: (BN, BM)
  // Shared: 2 * BM * BK * sizeof(float)
  ```
- Prefer `constexpr` for tile sizes, avoid magic numbers
- All kernels must handle edge cases: non-power-of-two dimensions, seq_len < block_size, batch_size=1

### Python

- Follow `black` formatting (`pip install black && black .`)
- Type hints on all public API functions
- Keep `python/binding.cpp` minimal — just the pybind11 glue, no logic

### Commit messages

Use conventional commits:
```
fix: resolve numerical overflow in softmax kernel for long sequences
feat: add FP16 attention kernel with 2x memory reduction
bench: add GQA benchmark vs standard MHA
docs: clarify tiling strategy in gemm.cu comments
```

---

## Reporting bugs

Open a GitHub issue with:

1. **Environment**: GPU model, CUDA version, OS, Python version
2. **Reproduction steps**: minimal code to trigger the bug
3. **Expected vs actual output**: if it's a numerical issue, include the max absolute error and which kernel is involved
4. **Nsight profile** (if available): attach a `.ncu-rep` or screenshot from Nsight Compute if it's a performance regression

---

## Questions

If you're unsure whether something is worth contributing, open a [Discussion](https://github.com/lecuong1502/CUDA-Inference-Engine/discussions) rather than an issue. Questions about the kernel implementation, GPU architecture, or quantization math are also welcome there.

---

## License

By contributing, you agree that your contributions will be licensed under the [Apache 2.0 License](./LICENSE) — same as the rest of the project.
