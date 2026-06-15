# Profiling harness for kernels
# Measures GFLOPS, bandwidth, latency — compares vs PyTorch (cuBLAS) baseline
#
# Usage:
#   python benchmarks/run_benchmark.py              # all benchmarks
#   python benchmarks/run_benchmark.py --kernel gemm
#   python benchmarks/run_benchmark.py --save-plot  # save plots/bench_phase1.png


import argparse
import time
import sys
import os

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../build"))
try:
    import cuda_engine as engine
    CUDA_AVAILABLE = True
except ImportError:
    print("[WARN] cuda_engine not found — benchmarking PyTorch baseline only")
    CUDA_AVAILABLE = False
 
WARMUP  = 50
RUNS    = 500
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

# ========================================
# Timer utilities
# ========================================

def cuda_time_ms(fn, warmup=WARMUP, runs=RUNS):
    """Return median latency in ms using CUDA events."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(runs):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return float(np.median(times))

# ========================================
# GEMM benchmark
# ========================================

def bench_gemm():
    print("\n== GEMM ====================================")
    print(f"{'Shape (M,N,K)':<22} {'Backend':<14} {'ms':>6} {'GFLOPS':>8} {'% cuBLAS':>10}")
    print("-" * 64)

    sizes = [
        (64,   64,   64),
        (256,  256,  256),
        (512,  512,  512),
        (768,  768,  768),    # GPT-2 small hidden
        (1024, 1024, 1024),
        (1,    768,  768),    # inference: single token projection
        (32,   768,  768),    # inference: batch=32
    ]

    for M, N, K in sizes:
        A = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
        B = torch.randn(K, N, device=DEVICE, dtype=torch.float32)
 
        flops = 2 * M * N * K  # multiply-add counts as 2
 
        # PyTorch (cuBLAS)
        t_cublas = cuda_time_ms(lambda: torch.mm(A, B))
        gflops_cublas = flops / t_cublas / 1e6  # GFLOPS

        label = f"({M},{N},{K})"
 
        if CUDA_AVAILABLE:
            C = torch.empty(M, N, device=DEVICE)
            t_engine = cuda_time_ms(lambda: engine.matmul(A, B))
            gflops_engine = flops / t_engine / 1e6
            pct = gflops_engine / gflops_cublas * 100
 
            print(f"{label:<22} {'cuBLAS':<14} {t_cublas:>6.3f} {gflops_cublas:>8.1f}")
            print(f"{'':22} {'engine':<14} {t_engine:>6.3f} {gflops_engine:>8.1f} {pct:>9.1f}%")
        else:
            print(f"{label:<22} {'cuBLAS':<14} {t_cublas:>6.3f} {gflops_cublas:>8.1f}")
 
        print()

# ========================================
# Softmax benchmark
# ========================================

def bench_softmax():
    print("\n== Softmax =====================================")
    print(f"{'Shape (rows, cols)':<24} {'Backend':<14} {'ms':>6} {'GB/s':>8}")
    print("-" * 56)

    sizes = [
        (32,   512),
        (32,   1024),
        (64,   512),
        (512,  512),    # attention matrix 512x512
        (1,    50257),  # next-token softmax over vocab
    ]
 
    for rows, cols in sizes:
        x = torch.randn(rows, cols, device=DEVICE, dtype=torch.float32)
        bytes_io = 2 * x.numel() * 4   # read + write, float32
 
        # PyTorch baseline
        t_pt = cuda_time_ms(lambda: F.softmax(x, dim=-1))
        bw_pt = bytes_io / t_pt / 1e6  # GB/s
 
        label = f"({rows}, {cols})"
 
        if CUDA_AVAILABLE:
            t_engine = cuda_time_ms(lambda: engine.softmax(x))
            bw_engine = bytes_io / t_engine / 1e6
 
            print(f"{label:<24} {'PyTorch':<14} {t_pt:>6.3f} {bw_pt:>8.1f}")
            print(f"{'':24} {'engine':<14} {t_engine:>6.3f} {bw_engine:>8.1f}")
        else:
            print(f"{label:<24} {'PyTorch':<14} {t_pt:>6.3f} {bw_pt:>8.1f}")
 
        print()

# ========================================
# Layer Norm benchmark
# ========================================

def bench_layernorm():
    print("\n== Layer norm ========================================")
    print(f"{'Shape (rows, hidden)':<24} {'Backend':<14} {'ms':>6} {'GB/s':>8}")
    print("-" * 56)

    sizes = [
        (32,  768),
        (64,  768),
        (128, 768),
        (32,  1024),
        (512, 768),
    ]
 
    eps = 1e-5
 
    for rows, hidden in sizes:
        x = torch.randn(rows, hidden, device=DEVICE, dtype=torch.float32)
        gamma = torch.ones(hidden,  device=DEVICE, dtype=torch.float32)
        beta = torch.zeros(hidden, device=DEVICE, dtype=torch.float32)
 
        ln = torch.nn.LayerNorm(hidden, eps=eps).to(DEVICE)
        bytes_io = 2 * x.numel() * 4
 
        # PyTorch baseline
        t_pt = cuda_time_ms(lambda: ln(x))
        bw_pt = bytes_io / t_pt / 1e6
 
        label = f"({rows}, {hidden})"
 
        if CUDA_AVAILABLE:
            t_engine = cuda_time_ms(lambda: engine.layernorm(x, gamma, beta, eps))
            bw_engine = bytes_io / t_engine / 1e6
 
            print(f"{label:<24} {'PyTorch':<14} {t_pt:>6.3f} {bw_pt:>8.1f}")
            print(f"{'':24} {'engine':<14} {t_engine:>6.3f} {bw_engine:>8.1f}")
        else:
            print(f"{label:<24} {'PyTorch':<14} {t_pt:>6.3f} {bw_pt:>8.1f}")
 
        print()

# ========================================
# Summary Table
# ========================================

def print_env():
    print("=" * 64)
    print("  CUDA Inference Engine — Phase 1 benchmark")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU:    {props.name}")
        print(f"  VRAM:   {props.total_memory / 1024**3:.1f} GB")
        print(f"  SMs:    {props.multi_processor_count}")
    print(f"  Warmup: {WARMUP} runs  |  Measured: {RUNS} runs  |  Stat: median")
    print("=" * 64)

# ========================================
# Optional: Save plot
# ========================================

def save_plot(data: dict):
    """Generate a simple bar chart comparing engine vs PyTorch."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("[WARN] matplotlib not installed, skipping plot")
        return
 
    os.makedirs("benchmarks/plots", exist_ok=True)

    fig, axes = plt.subplots(1, len(data), figsize=(5 * len(data), 4))
    if len(data) == 1:
        axes = [axes]

    for ax, (title, values) in zip(axes, data.items()):
        labels = list(values.keys())
        metrics = list(values.values())
        colors = ["#378ADD" if "engine" in l else "#888780" for l in labels]
        bars = ax.bar(labels, metrics, color=colors, width=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("GFLOPS" if "GEMM" in title else "GB/s")
        ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("CUDA Inference Engine — Phase 1 kernels", fontsize=12, y=1.02)
    plt.tight_layout()
    out_path = "benchmarks/plots/bench_phase1.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved to {out_path}")

# ========================================
# Main
# ========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel",    choices=["gemm", "softmax", "layernorm", "all"],
                        default="all")
    parser.add_argument("--save-plot", action="store_true")
    args = parser.parse_args()
 
    if DEVICE == "cpu":
        print("[WARN] No CUDA device found — results will not reflect GPU performance")
 
    print_env()
 
    if args.kernel in ("gemm", "all"):
        bench_gemm()
    if args.kernel in ("softmax", "all"):
        bench_softmax()
    if args.kernel in ("layernorm", "all"):
        bench_layernorm()

if __name__ == "__main__":
    main()