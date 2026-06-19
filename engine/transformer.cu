// engine/transformer.cu
// Full GPT-2 transformer block: LayerNorm → MHA → residual → LayerNorm → FFN → residual
// One block = one decoder layer

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cassert>
#include <cstdio>

// Forward declarations from other kernel files
void launch_gemm(const float*, const float*, float*, int, int, int, cudaStream_t);
void launch_gemm_batched(const float*, const float*, float*, int, int, int, int, cudaStream_t);
void launch_layernorm(const float*, const float*, const float*, float*, int, int, float, cudaStream_t);
void launch_add_layernorm(const float*, const float*, const float*, const float*, float*, int, int, float, cudaStream_t);
void launch_flash_attention(const float*, const float*, const float*, float*, int, int, int, int, cudaStream_t);
void launch_gelu(const float*, float*, int, cudaStream_t);

// =============================================
// GPT-2 small config
// =============================================

struct GPT2Config {
    int vocab_size = 50257;
    int seq_len    = 1024;   // max context length
    int n_embd     = 768;    // hidden dim
    int n_layer    = 12;     // number of transformer blocks
    int n_head     = 12;     // attention heads
    int head_size  = 64;     // n_embd / n_head
    int ffn_dim    = 3072;   // 4 * n_embd
    float eps      = 1e-5f;
};

// ================================================
// Weights for a single transformer block
// ================================================

struct BlockWeights {
    // Layer norm 1 (before attention)
    float* ln1_gamma;   // (n_embd,)
    float* ln1_beta;

    // QKV projection: maps (n_embd) → (3 * n_embd)
    float* qkv_weight;  // (3 * n_embd, n_embd)
    float* qkv_bias;    // (3 * n_embd,)

    // Output projection: (n_embd, n_embd)
    float* attn_proj_weight;
    float* attn_proj_bias;

    // Layer norm 2 (before FFN)
    float* ln2_gamma;
    float* ln2_beta;

    // FFN: fc1 (n_embd → ffn_dim), fc2 (ffn_dim → n_embd)
    float* fc1_weight;  // (ffn_dim, n_embd)
    float* fc1_bias;    // (ffn_dim,)
    float* fc2_weight;  // (n_embd, ffn_dim)
    float* fc2_bias;    // (n_embd,)
};

// ===================================================================
// Scratch buffers for intermediate activations (allocated once)
// ===================================================================

struct BlockBuffers {
    float* ln1_out;       // (B*T, n_embd)
    float* qkv;           // (B*T, 3*n_embd)
    float* q, *k, *v;     // (B, H, T, HS) each
    float* attn_out;      // (B, H, T, HS)
    float* attn_merged;   // (B*T, n_embd)
    float* attn_proj;     // (B*T, n_embd)
    float* ln2_out;       // (B*T, n_embd)
    float* ffn_h;         // (B*T, ffn_dim)
    float* ffn_h_gelu;    // (B*T, ffn_dim)
    float* ffn_out;       // (B*T, n_embd)
};

// ======================================================
// Bias-add kernel (fused with GEMM output)
// ======================================================

__global__ void add_bias_kernel(
    float* __restrict__ x,
    const float* __restrict__ bias,
    int rows, int cols
) {
    int row = blockIdx.x;
    int col = threadIdx.x + blockIdx.y * blockDim.x;
    if (row < rows && col < cols)
        x[row * cols + col] += bias[col];
}

void launch_add_bias(float* x, const float* bias, int rows, int cols, cudaStream_t s = 0) {
    dim3 block(256);
    dim3 grid(rows, (cols + 255) / 256);
    add_bias_kernel<<<grid, block, 0, s>>>(x, bias, rows, cols);
}

// ===========================================================================
// Reshape + transpose: (B*T, 3*C) → Q(B,H,T,HS), K(B,H,T,HS), V(B,H,T,HS)
// ===========================================================================

__global__ void split_qkv_kernel(
    const float* __restrict__ qkv,   // (B*T, 3*C)
    float* __restrict__ Q,
    float* __restrict__ K,
    float* __restrict__ V,
    int B, int H, int T, int HS
) {
    int C = H * HS;
    int bt = blockIdx.x;             // flattened (b, t)
    int d = threadIdx.x;            // dimension index within 3*C

    if (bt >= B * T || d >= 3 * C) return;

    int b = bt / T;
    int t = bt % T;

    float val = qkv[bt * 3 * C + d];

    int part = d / C;               // 0=Q, 1=K, 2=V
    int hd = d % C;               // index within C
    int head = hd / HS;
    int dim = hd % HS;

    int out_idx = ((b * H + head) * T + t) * HS + dim;

    if (part == 0) Q[out_idx] = val;
    else if (part == 1) K[out_idx] = val;
    else V[out_idx] = val;
}

void launch_split_qkv(
    const float* qkv, float* Q, float* K, float* V,
    int B, int H, int T, int HS,
    cudaStream_t s = 0
) {
    int C = H * HS;
    dim3 block(min(3 * C, 1024));
    dim3 grid(B * T, (3 * C + 1023) / 1024);
    split_qkv_kernel<<<grid, block, 0, s>>>(qkv, Q, K, V, B, H, T, HS);
}

// =======================================================
// Merge attention output: (B,H,T,HS) → (B*T, C)
// =======================================================

__global__ void merge_heads_kernel(
    const float* __restrict__ attn,  // (B, H, T, HS)
    float* __restrict__ out,         // (B*T, C)
    int B, int H, int T, int HS
) {
    int C = H * HS;
    int bt = blockIdx.x;
    int d = threadIdx.x;

    if (bt >= B * T || d >= C) return;

    int b = bt / T;
    int t = bt % T;
    int head = d / HS;
    int dim = d % HS;

    int in_idx = ((b * H + head) * T + t) * HS + dim;
    out[bt * C + d] = attn[in_idx];
}

void launch_merge_heads(
    const float* attn, float* out,
    int B, int H, int T, int HS,
    cudaStream_t s = 0
) {
    int C = H * HS;
    dim3 block(min(C, 1024));
    dim3 grid(B * T, (C + 1023) / 1024);
    merge_heads_kernel<<<grid, block, 0, s>>>(attn, out, B, H, T, HS);
}

// ==============================
// Residual add kernel
// ==============================

__global__ void residual_add_kernel(
    float* __restrict__ x,         // in-place: x += residual
    const float* __restrict__ res,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] += res[idx];
}

void launch_residual_add(float* x, const float* res, int n, cudaStream_t s = 0) {
    int block = 256;
    int grid  = (n + block - 1) / block;
    residual_add_kernel<<<grid, block, 0, s>>>(x, res, n);
}

// ===================================================================
// Transformer block forward pass
// input:  (B*T, n_embd) — token embeddings for this layer
// output: (B*T, n_embd) — updated embeddings (in-place into `input`)
// ===================================================================

void transformer_block_forward(
    float*  input,             // (B*T, C) — modified in-place
    const BlockWeights& w,
    BlockBuffers& buf,
    const GPT2Config& cfg,
    int B, int T,
    cudaStream_t stream = 0
) {
    int C = cfg.n_embd;
    int H = cfg.n_head;
    int HS = cfg.head_size;
    int FFN = cfg.ffn_dim;
    int BT = B * T;

    // LayerNorm 1
    launch_layernorm(input, w.ln1_gamma, w.ln1_beta,
                     buf.ln1_out, BT, C, cfg.eps, stream);

    // QKV projection: (BT, C) × (C, 3C)ᵀ → (BT, 3C)
    launch_gemm(buf.ln1_out, w.qkv_weight, buf.qkv, BT, 3*C, C, stream);
    launch_add_bias(buf.qkv, w.qkv_bias, BT, 3*C, stream);

    // Split QKV → (B, H, T, HS)
    launch_split_qkv(buf.qkv, buf.q, buf.k, buf.v, B, H, T, HS, stream);

    // Flash Attention
    launch_flash_attention(buf.q, buf.k, buf.v, buf.attn_out,
                           B, H, T, HS, stream);

    // Merge heads: (B, H, T, HS) → (BT, C)
    launch_merge_heads(buf.attn_out, buf.attn_merged, B, H, T, HS, stream);

    // Output projection: (BT, C) × (C, C)ᵀ → (BT, C)
    launch_gemm(buf.attn_merged, w.attn_proj_weight, buf.attn_proj,
                BT, C, C, stream);
    launch_add_bias(buf.attn_proj, w.attn_proj_bias, BT, C, stream);

    // Residual add: input = input + attn_proj
    launch_residual_add(input, buf.attn_proj, BT * C, stream);

    // LayerNorm 2
    launch_layernorm(input, w.ln2_gamma, w.ln2_beta,
                     buf.ln2_out, BT, C, cfg.eps, stream);

    // FFN fc1: (BT, C) × (C, FFN)ᵀ → (BT, FFN)
    launch_gemm(buf.ln2_out, w.fc1_weight, buf.ffn_h, BT, FFN, C, stream);
    launch_add_bias(buf.ffn_h, w.fc1_bias, BT, FFN, stream);

    // GELU activation
    launch_gelu(buf.ffn_h, buf.ffn_h_gelu, BT * FFN, stream);

    // FFN fc2: (BT, FFN) × (FFN, C)ᵀ → (BT, C)
    launch_gemm(buf.ffn_h_gelu, w.fc2_weight, buf.ffn_out, BT, C, FFN, stream);
    launch_add_bias(buf.ffn_out, w.fc2_bias, BT, C, stream);

    // Residual add: input = input + ffn_out
    launch_residual_add(input, buf.ffn_out, BT * C, stream);
}

// ===============================
// Buffer allocation helper
// ===============================

BlockBuffers alloc_block_buffers(const GPT2Config& cfg, int B, int T) {
    BlockBuffers buf;
    int C = cfg.n_embd;
    int H = cfg.n_head;
    int HS = cfg.head_size;
    int FFN = cfg.ffn_dim;
    int BT = B * T;
    int BHTHS = B * H * T * HS;

    auto alloc = [](float** ptr, size_t n) {
        cudaMalloc(ptr, n * sizeof(float));
    };

    alloc(&buf.ln1_out, BT * C);
    alloc(&buf.qkv, BT * 3 * C);
    alloc(&buf.q, BHTHS);
    alloc(&buf.k, BHTHS);
    alloc(&buf.v, BHTHS);
    alloc(&buf.attn_out, BHTHS);
    alloc(&buf.attn_merged, BT * C);
    alloc(&buf.attn_proj, BT * C);
    alloc(&buf.ln2_out, BT * C);
    alloc(&buf.ffn_h, BT * FFN);
    alloc(&buf.ffn_h_gelu, BT * FFN);
    alloc(&buf.ffn_out, BT * C);

    return buf;
}

void free_block_buffers(BlockBuffers& buf) {
    cudaFree(buf.ln1_out);
    cudaFree(buf.qkv);
    cudaFree(buf.q);
    cudaFree(buf.k);
    cudaFree(buf.v);
    cudaFree(buf.attn_out);
    cudaFree(buf.attn_merged);
    cudaFree(buf.attn_proj);
    cudaFree(buf.ln2_out);
    cudaFree(buf.ffn_h);
    cudaFree(buf.ffn_h_gelu);
    cudaFree(buf.ffn_out);
}