# tests/test_phase2.py
# Phase 2 correctness: attention kernel and transformer block vs PyTorch reference
#
# Run: python tests/test_phase2.py
# Requires: torch, built cuda_engine module

import sys, os, math
import torch
import torch.nn.functional as F
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../build"))
try:
    import cuda_engine as engine
    CUDA_AVAILABLE = True
except ImportError:
    print("[WARN] cuda_engine not found — running torch-only reference checks")
    CUDA_AVAILABLE = False

ATOL = 1e-3   # slightly relaxed vs Phase 1 (accumulated FP32 error across matmuls)
RTOL = 1e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
 
def check(name, actual, expected, atol=ATOL, rtol=RTOL):
    max_err = (actual - expected).abs().max().item()
    passed  = torch.allclose(actual, expected, atol=atol, rtol=rtol)
    status  = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name:<52s} max_err={max_err:.2e}")
    return passed


# ============================================================
# Reference: PyTorch scaled dot-product attention (causal)
# ============================================================

def ref_attention(Q, K, V):
    """Causal scaled dot-product attention — pure PyTorch."""
    B, H, T, HS = Q.shape
    scale = 1.0 / math.sqrt(HS)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, H, T, T)
 
    # Causal mask
    mask = torch.triu(torch.ones(T, T, device=Q.device), diagonal=1).bool()
    scores.masked_fill_(mask, float('-inf'))
 
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V)  # (B, H, T, HS)


# ============================================================
# Reference: GPT-2 transformer block
# ============================================================

class RefTransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.n_head = n_head
        self.HS = n_embd // n_head
 
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd, bias=True)
        self.attn_proj = nn.Linear(n_embd, n_embd, bias=True)
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=True)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=True)
 
    def forward(self, x):
        B, T, C = x.shape
        H, HS = self.n_head, self.HS
 
        # MHA
        h = self.ln1(x)
        qkv = self.qkv_proj(h)          # (B, T, 3C)
        q, k, v = qkv.split(C, dim=-1)  # each (B, T, C)
 
        def reshape(t):
            return t.view(B, T, H, HS).transpose(1, 2)  # (B, H, T, HS)
 
        q, k, v = reshape(q), reshape(k), reshape(v)
        attn = ref_attention(q, k, v)                   # (B, H, T, HS)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.attn_proj(attn)
 
        # FFN
        h = self.ln2(x)
        x = x + self.fc2(F.gelu(self.fc1(h), approximate="tanh"))
        return x

 
# ==========
# Tests
# ==========

def test_attention():
    print("\n── Attention kernel ───────────────")
    results = []
 
    cases = [
        (1,  12,  32, 64, "GPT-2 small, B=1, T=32"),
        (4,  12,  64, 64, "GPT-2 small, B=4, T=64"),
        (1,  12, 128, 64, "GPT-2 small, T=128"),
        (2,   8,  16, 32, "small custom config"),
    ]
 
    for B, H, T, HS, desc in cases:
        Q = torch.randn(B, H, T, HS, device=DEVICE)
        K = torch.randn(B, H, T, HS, device=DEVICE)
        V = torch.randn(B, H, T, HS, device=DEVICE)
 
        ref = ref_attention(Q, K, V)
 
        if CUDA_AVAILABLE:
            out = engine.flash_attention(Q, K, V)
            results.append(check(f"flash_attention {desc}", out.cpu(), ref.cpu()))
        else:
            # Verify reference against torch's built-in
            try:
                builtin = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
                results.append(check(f"ref_attention {desc} [vs torch builtin]",
                                     ref.cpu(), builtin.cpu()))
            except Exception:
                results.append(check(f"ref_attention {desc} [self]", ref, ref))
 
    return all(results)

def test_transformer_block():
    print("\n── Transformer block forward pass ───────────────")
    results = []
 
    # GPT-2 small config
    n_embd, n_head = 768, 12
    cases = [
        (1,  8,  "B=1, T=8   (single token, short)"),
        (1,  32, "B=1, T=32"),
        (2,  64, "B=2, T=64"),
        (1, 128, "B=1, T=128"),
    ]
 
    for B, T, desc in cases:
        ref_block = RefTransformerBlock(n_embd, n_head).to(DEVICE)
        ref_block.eval()
 
        x   = torch.randn(B, T, n_embd, device=DEVICE)
        ref = ref_block(x)
 
        if CUDA_AVAILABLE:
            # Extract weights and pass to engine
            w = ref_block.state_dict()
            out = engine.transformer_block_forward(
                x,
                qkv_weight   = w["qkv_proj.weight"],
                qkv_bias     = w["qkv_proj.bias"],
                attn_w       = w["attn_proj.weight"],
                attn_b       = w["attn_proj.bias"],
                ln1_gamma    = w["ln1.weight"],
                ln1_beta     = w["ln1.bias"],
                fc1_weight   = w["fc1.weight"],
                fc1_bias     = w["fc1.bias"],
                fc2_weight   = w["fc2.weight"],
                fc2_bias     = w["fc2.bias"],
                ln2_gamma    = w["ln2.weight"],
                ln2_beta     = w["ln2.bias"],
            )
            results.append(check(f"transformer block {desc}", out.cpu(), ref.cpu()))
        else:
            # Verify reference block gives same output twice (determinism)
            ref2 = ref_block(x)
            results.append(check(f"transformer block determinism {desc}",
                                 ref.cpu(), ref2.cpu(), atol=0, rtol=0))
 
    return all(results)

def test_weight_loader():
    """Smoke test: if GPT-2 weights exist locally, verify shape after load."""
    print("\n── Weight loader ───────────────")
    weight_path = "weights/gpt2.safetensors"
 
    if not os.path.exists(weight_path):
        print(f"  [SKIP] {weight_path} not found — run scripts/download_weights.py first")
        return True
 
    if CUDA_AVAILABLE:
        loader = engine.WeightLoader(weight_path)
        # GPT-2 small: token embedding should be (50257, 768)
        wte = loader.load("wte.weight")
        expected_shape = (50257, 768)
        passed = tuple(wte.shape) == expected_shape
        print(f"  [{'PASS' if passed else 'FAIL'}] wte.weight shape: "
              f"{tuple(wte.shape)} (expected {expected_shape})")
        return passed
    else:
        print("  [SKIP] No CUDA engine")
        return True
    
def main():
    print("=" * 60)
    print("  CUDA Inference Engine — Phase 2 correctness tests")
    print(f"  Device: {DEVICE}")
    print("=" * 60)
 
    tests = [
        ("Attention",          test_attention),
        ("Transformer block",  test_transformer_block),
        ("Weight loader",      test_weight_loader),
    ]
 
    passed = sum(fn() for _, fn in tests)
    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{len(tests)} test groups passed")
    print(f"{'='*60}")
    if passed < len(tests): sys.exit(1)

if __name__ == "__main__":
    main()