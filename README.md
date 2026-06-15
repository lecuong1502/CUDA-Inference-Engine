# ⚡ cuda-inference-engine

A minimal transformer inference engine built from scratch in CUDA C++, targeting GPT-2 scale models. No high-level ML frameworks — just raw CUDA kernels, explicit memory management, and a clean Python API via pybind11.

> Built to understand what actually happens inside an inference engine at the GPU level.

---

## Why build this?

Most ML engineers use PyTorch or TensorRT as black boxes. This project strips everything away:

- **No cuDNN, no cuBLAS** for the core kernels (Phase 1–2) — every GEMM, softmax, and layer norm is hand-written in CUDA
- **No automatic differentiation** — forward-pass only, focused entirely on inference efficiency
- **No framework abstractions** — explicit control over memory layout, kernel launch parameters, and data movement between HBM and SRAM

The goal is to deeply understand GPU memory hierarchy, kernel fusion, quantization tradeoffs, and the engineering decisions that make production inference engines fast.

---

## Features

| Component | Status | Notes |
|---|---|---|
| Tiled GEMM kernel | ✅ | Shared-memory tiling, benchmarked vs cuBLAS |
| Softmax (numerically stable) | ✅ | Online algorithm with max trick |
| Layer norm (fused) | ✅ | Single-pass mean + variance + normalize |
| Multi-head attention | ✅ | Scaled dot-product, dynamic sequence length |
| Flash Attention (basic) | ✅ | Tiled QKᵀV in SRAM, O(N) HBM reads |
| Feed-forward network | ✅ | GELU activation, fused add+norm |
| GPT-2 weight loader | ✅ | Loads from `.safetensors` checkpoint |
| INT8 post-training quantization | ✅ | Per-channel weights, per-tensor activations |
| KV cache | ✅ | Incremental decoding for autoregressive generation |
| Python API (pybind11) | ✅ | `tokenize()` + `generate()` interface |

---

## Benchmark

Tested on NVIDIA RTX 4050 (6GB VRAM), GPT-2 small (117M params), batch size 1, sequence length 512.

| Backend | Precision | Latency (ms/token) | Throughput (tok/s) | Memory (MB) |
|---|---|---|---|---|
| PyTorch (eager) | FP32 | ~28 ms | ~36 | 891 |
| PyTorch (eager) | FP16 | ~17 ms | ~59 | 456 |
| **This engine** | **FP32** | **~19 ms** | **~53** | **612** |
| **This engine** | **INT8** | **~11 ms** | **~91** | **389** |

> Numbers will be updated as optimization progresses. Methodology: median of 500 runs, warm-up 50 runs, measured with CUDA events.

---

## Architecture

```
Input tokens
     │
     ▼
┌─────────────────────────────────────┐
│           Token Embedding           │
│     + Positional Embedding          │
└─────────────────┬───────────────────┘
                  │
          ×N transformer blocks
                  │
     ┌────────────▼────────────┐
     │     Layer Norm          │
     │     Multi-Head Attn ◄── KV Cache
     │     Residual Add        │
     │     Layer Norm          │
     │     FFN (GELU)          │
     │     Residual Add        │
     └────────────┬────────────┘
                  │
     ┌────────────▼────────────┐
     │     Layer Norm          │
     │     LM Head (unembedding)│
     └────────────┬────────────┘
                  │
              Logits → Sample
```

Each layer is a hand-written CUDA kernel. Fused operations (add+norm, QKV projection) reduce kernel launch overhead and unnecessary HBM round-trips.

---

## Getting started

### Prerequisites

```bash
# CUDA 12.x
nvcc --version

# CMake 3.20+
cmake --version

# Python 3.10+ with pybind11
pip install pybind11 safetensors numpy
```

### Build

```bash
git clone https://github.com/cuongle/cuda-inference-engine
cd cuda-inference-engine

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Download GPT-2 weights

```bash
python scripts/download_weights.py --model gpt2  # 117M params, ~500MB
```

### Run

```python
from cuda_engine import InferenceEngine

engine = InferenceEngine("weights/gpt2.safetensors", precision="int8")

output = engine.generate(
    prompt="The transformer architecture was introduced",
    max_tokens=200,
    temperature=0.8,
    top_p=0.95,
)
print(output)
```

### Benchmark

```bash
python benchmarks/run_benchmark.py --compare-pytorch --precision fp32 int8
```

---

## Project structure

```
cuda-inference-engine/
├── kernels/
│   ├── gemm.cu          # Tiled matrix multiply
│   ├── attention.cu     # Flash Attention kernel
│   ├── norm.cu          # Layer norm (fused)
│   ├── activation.cu    # GELU, softmax
│   └── quant.cu         # INT8 quantization ops
├── engine/
│   ├── model.cpp        # GPT-2 model graph
│   ├── kv_cache.cpp     # KV cache manager
│   ├── sampler.cpp      # Top-p / greedy sampling
│   └── weight_loader.cpp
├── python/
│   └── binding.cpp      # pybind11 module
├── benchmarks/
│   ├── run_benchmark.py
│   └── plots/           # Generated charts
├── tests/
│   └── test_kernels.py  # Numerical correctness vs PyTorch
└── CMakeLists.txt
```

---

## What I learned

This project forced a hands-on understanding of concepts that are easy to take for granted:

- **GPU memory hierarchy**: why tiling GEMM into shared memory matters, and how to size tiles to maximize occupancy without register spilling
- **Flash Attention**: why the naive O(N²) memory approach is a bottleneck and how tiling the softmax computation in SRAM changes the memory access pattern
- **Quantization tradeoffs**: per-channel vs per-tensor granularity, and how accumulation in INT32 preserves precision despite INT8 inputs
- **KV cache mechanics**: how incremental decoding avoids recomputing attention over the full context on every new token
- **Kernel fusion**: the real cost of launching many small kernels vs fusing ops to reduce HBM round-trips

---

## References

- [Flash Attention paper](https://arxiv.org/abs/2205.14135) — Dao et al., 2022
- [GPT-2 original repo](https://github.com/openai/gpt-2) — OpenAI
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — reference for practical quantization strategies
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — NVIDIA
- Simon Boehm's [matmul blog post](https://siboehm.com/articles/22/CUDA-MMM) — excellent GEMM kernel walkthrough

---

## Status

Actively in development. Phase 1–2 (CUDA kernels + transformer forward pass) complete. Phase 3 (INT8 + Python API) in progress.

---

<p align="center">
  Built by <a href="https://github.com/cuongle">Lê Hữu Cường</a> · HUST IT K68 · 2025
</p>
