# Numerical correctness tests — CUDA kernel output vs PyTorch reference
# Run after building: python tests/test_kernels.py
#
# Requires:
#   pip install torch numpy
#   Built pybind11 module in build/
#   (or adjust sys.path below)

import sys
import os
import math
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../build"))

try:
    import cuda_engine as engine
    CUDA_AVAILABLE = True
except ImportError:
    print("[WARN] cuda_engine module not found — running reference checks only")
    CUDA_AVAILABLE = False

ATOL = 1e-4
RTOL = 1e-3

# ==========================================
# Helpers
# ==========================================

def check(name, actual, expected, atol=ATOL, rtol=RTOL):
    """Assert closeness, print pass/fail with max error."""
    max_err = (actual - expected).abs().max().item()
    passed  = torch.allclose(actual, expected, atol=atol, rtol=rtol)
    status  = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name:<40s}  max_err={max_err:.2e}")
    if not passed:
        # Show where the biggest error is
        idx = (actual - expected).abs().argmax()
        print(f"         worst: got {actual.flatten()[idx].item():.6f}, "
              f"expected {expected.flatten()[idx].item():.6f}")
    return passed

# ==========================================
# GEMM
# ==========================================

def test_gemm():
    print("\n== GEMM ==========================================")
    results = []

    cases = [
        (64,  64,  64,  "square small"),
        (128, 256, 512, "non-square"),
        (33,  65,  97,  "non-power-of-2"),
        (512, 512, 512, "square large"),
        (1,   768, 768, "batch=1 (inference)"),
    ]

    for M, N, K, desc in cases:
        A = torch.randn(M, K, dtype=torch.float32)
        B = torch.randn(K, N, dtype=torch.float32)
        ref = A @ B

        if CUDA_AVAILABLE:
            out = engine.matmul(A.cuda(), B.cuda()).cpu()
            results.append(check(f"GEMM {M}x{K}x{N} ({desc})", out, ref))
        else:
            # Self-check: numpy vs torch
            np_out = torch.from_numpy(A.numpy() @ B.numpy())
            results.append(check(f"GEMM {M}x{K}x{N} ({desc}) [numpy ref]", np_out, ref))

    return all(results)

# ==========================================
# Softmax
# ==========================================

def test_softmax():
    print("\n== Softmax ===================================")
    results = []

    cases = [
        (32,  128,  "standard"),
        (64,  512,  "seq=512"),
        (1,   50257,"vocab (GPT-2)"),     # softmax over vocab for next-token
        (32,  1000, "non-power-of-2"),
    ]

    for rows, cols, desc in cases:
        x   = torch.randn(rows, cols, dtype=torch.float32)
        ref = F.softmax(x, dim=-1)
 
        if CUDA_AVAILABLE:
            out = engine.softmax(x.cuda()).cpu()
            results.append(check(f"softmax {rows}x{cols} ({desc})", out, ref))
        else:
            # Verify the numerically stable formula manually
            x_max  = x.max(dim=-1, keepdim=True).values
            exp_x  = (x - x_max).exp()
            manual = exp_x / exp_x.sum(dim=-1, keepdim=True)
            results.append(check(f"softmax {rows}x{cols} ({desc}) [manual stable]", manual, ref))

     # Numerical stability test: large values that would overflow naive exp
    print("\n  Numerical stability:")
    x_big = torch.full((1, 64), 1000.0)
    ref   = F.softmax(x_big, dim=-1)   # all 1/64
 
    if CUDA_AVAILABLE:
        out = engine.softmax(x_big.cuda()).cpu()
        results.append(check("softmax large values (stability)", out, ref))
    else:
        x_max  = x_big.max(dim=-1, keepdim=True).values
        manual = (x_big - x_max).exp()
        manual = manual / manual.sum(dim=-1, keepdim=True)
        results.append(check("softmax large values (stability) [manual]", manual, ref))
 
    return all(results)

# ==========================================
# Layer Norm
# ==========================================

def test_layernorm():
    print("\n== Layer Norm =================================")
    results = []

    cases = [
        (32,  768,  "GPT-2 small hidden"),
        (1,   768,  "batch=1"),
        (128, 1024, "GPT-2 medium hidden"),
        (64,  256,  "small"),
    ]
 
    eps = 1e-5
 
    for rows, hidden, desc in cases:
        x = torch.randn(rows, hidden, dtype=torch.float32)
        gamma = torch.ones(hidden,  dtype=torch.float32)
        beta = torch.zeros(hidden, dtype=torch.float32)
 
        ref_ln = torch.nn.LayerNorm(hidden, eps=eps)
        ref_ln.weight.data = gamma
        ref_ln.bias.data = beta
        ref = ref_ln(x)
 
        if CUDA_AVAILABLE:
            out = engine.layernorm(x.cuda(), gamma.cuda(), beta.cuda(), eps).cpu()
            results.append(check(f"layernorm {rows}x{hidden} ({desc})", out, ref))
        else:
            # Manual reference
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            norm = (x - mean) / (var + eps).sqrt()
            manual = gamma * norm + beta
            results.append(check(f"layernorm {rows}x{hidden} ({desc}) [manual]", manual, ref))
 
    # Learned gamma/beta
    print("\n  Learned gamma/beta:")
    rows, hidden = 16, 768
    x = torch.randn(rows, hidden)
    gamma = torch.randn(hidden)
    beta = torch.randn(hidden)
 
    ref_ln = torch.nn.LayerNorm(hidden, eps=eps)
    ref_ln.weight.data = gamma
    ref_ln.bias.data = beta
    ref = ref_ln(x)

    if CUDA_AVAILABLE:
        out = engine.layernorm(x.cuda(), gamma.cuda(), beta.cuda(), eps).cpu()
        results.append(check("layernorm learned gamma/beta", out, ref))
    else:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        manual = gamma * (x - mean) / (var + eps).sqrt() + beta
        results.append(check("layernorm learned gamma/beta [manual]", manual, ref))
 
    return all(results)

# ==========================================
# GELU
# ==========================================

def test_gelu():
    print("\n== GELU =================================")
    results = []

    cases = [
        (1024,  "small"),
        (49152, "GPT-2 FFN dim (768*4*16)"),
    ]
 
    for n, desc in cases:
        x = torch.randn(n, dtype=torch.float32)
        ref = F.gelu(x, approximate="tanh")
 
        if CUDA_AVAILABLE:
            out = engine.gelu(x.cuda()).cpu()
            results.append(check(f"gelu n={n} ({desc})", out, ref))
        else:
            c = math.sqrt(2 / math.pi)
            manual = 0.5 * x * (1 + torch.tanh(c * (x + 0.044715 * x**3)))
            results.append(check(f"gelu n={n} ({desc}) [manual]", manual, ref))
 
    return all(results)

# ==========================================
# Main
# ==========================================

def main():
    print("=" * 56)
    print("CUDA Inference Engine — kernel correctness tests")
    print("Mode:", "CUDA engine" if CUDA_AVAILABLE else "reference only (no CUDA module)")
    print("=" * 56)

    tests = [
        ("GEMM",       test_gemm),
        ("Softmax",    test_softmax),
        ("Layer norm", test_layernorm),
        ("GELU",       test_gelu),
    ]

    passed = 0
    for name, fn in tests:
        ok = fn()
        if ok:
            passed += 1

    print("\n" + "=" * 56)
    print(f"  Results: {passed}/{len(tests)} test groups passed")
    print("=" * 56)
 
    if passed < len(tests):
        sys.exit(1)

if __name__ == "__main__":
    main()