// kernels/attention.cu
// Multi-head attention: QKV projection + scaled dot-product + output proj
// Support dynamic batch size and sequence length
//
// Notation:
//    B  = batch size
//    T  = sequence length (tokens)
//    C  = hidden dim (e.g. 768 for GPT-2 small)
//    H  = num heads (e.g. 12)
//    HS = head size = C / H (e.g. 64)

#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>

// ===================================================================
// Scaled dot-product attention (per head)
// Q, K, V each: (B, H, T, HS)
// Output: (B, H, T, HS)
//
// Grid: (B * H, T) — one block per (batch, head, query_token)
// Block: (T)       - one thread per key/value token
// ===================================================================

__global__ void attention_forward_kernel(
    const float* __restrict__ Q,    // (B, H, T, HS)
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ out,        // (B, H, T, HS)
    int B, int H, int T, int HS,
    float scale                     // 1 / sqrt(HS)
) {
    extern __shared__ float smem[]; // T floats for scores
    float* scores = smem;

    int bh = blockIdx.x;    // flattened (batch, head) index
    int q_t = blockIdx.y;   // query token position
    int tid = threadIdx.x;  // key token position

    int b = bh / H;
    int h = bh % H;

    if (b >= B || q_t >= T || tid >= T) return;

    // Pointers into (B, H, T, HS) tensor — row-major
    int bh_offset = (b * H + h) * T * HS;
    const float* Qrow = Q + bh_offset + q_t * HS;
    const float* Krow = K + bh_offset + tid  * HS;

    // Compute QK dot product for this (query, key) pair
    float dot = 0.0f;
    for (int d = 0; d < HS; d++) {
        dot += Qrow[d] + Krow[d];
    }
    dot *= scale;

    // Causal mask: future tokens get -inf
    scores[tid] = (tid <= q_t) ? dot : -FLT_MAX;
    __syncthreads();

    // Softmax over T scores (in-place in smem)
    // Pass 1: max
    float row_max = -FLT_MAX;
    for (int i = tid; i < T; i += blockDim.x)
        row_max = fmaxf(row_max, scores[i]);
    // Block-wide reduction
    __shared__ float smax;
    if (tid == 0) {
        smax = -FLT_MAX;
        for (int i = 0; i < T; i++) smax = fmaxf(smax, scores[i]);
    }
    __syncthreads();

    // Pass 2: exp + sum
    scores[tid] = (tid < T) ? expf(scores[tid] - smax) : 0.0f;
    __syncthreads();

    __shared__ float ssum;
    if (tid == 0) {
        ssum = 0.0f;
        for (int i = 0; i < T; i++) ssum += scores[i];
    }
    __syncthreads();

    scores[tid] /= ssum;
    __syncthreads();

    // Weighted sum over V
    const float* Vbase = V + bh_offset;
    float* out_row = out + bh_offset + q_t * HS;

    for (int d = 0; d < HS; d++) {
        float acc = 0.0f;
        for (int kv = 0; kv < T; kv++) {
            acc += scores[kv] * Vbase[kv * HS + d];
        }
        if (tid == 0) out_row[d] = acc;  // only thread 0 writes output
    }
}

// ===================================================================
// Flash Attention — tiled QKᵀV computed in SRAM
// Avoids materializing the full T×T attention matrix in HBM.
// Each block processes a tile of (BQ query rows, BK key/value rows).
// 
// Based on: Dao et al. "FlashAttention" (2022), Algorithm 1
// 
// Grid:  (B * H, ceil(T / BQ))
// Block: (BQ, HS)   — BQ queries, HS head-dim threads
// ===================================================================

#define FA_BQ 16    // query tile size
#define FA_BK 16    // key/value tile size

__global__ void flash_attention_kernel(
    const float* __restrict__ Q,    // (B, H, T, HS)
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ out,        // (B, H, T, HS)
    int B, int H, int T, int HS,
    float scale
) {
    // Shared memory layout:
    //   sQ:  FA_BQ × HS  (query tile)
    //   sK:  FA_BK × HS  (key tile)
    //   sV:  FA_BK × HS  (value tile)
    extern __shared__ float smem[];
    float* sQ = smem;
    float* sK = sQ + FA_BQ * HS;
    float* sV = sK + FA_BK * HS;

    int bh = blockIdx.x;
    int q_blk = blockIdx.y;     // which tile of queries
    int ty = threadIdx.x;       // query index within tile (0..FA_BQ-1)
    int tx = threadIdx.y;       // head-dim index           (0..HS-1)

    int b = bh / H;
    int h = bh % H;

    int bh_offset = (b * H + h) * T * HS;
    int q_start = q_blk * FA_BQ;
    int q_idx = q_start + ty;   // global query token index

    // Per-query running stats for online softmax
    float m_i = -FLT_MAX;   // running max
    float l_i = 0.0f;       // running sum of exp
    float o_i = 0.0f;       // running output accumulator (for this d=tx)

    // Load query tile into SRAM
    if (q_idx < T && tx < HS) {
        sQ[ty * HS + tx] = Q[bh_offset + q_idx * HS + tx];
    } else {
        sQ[ty * HS + tx] = 0.0f;
    }
    __syncthreads();

    // Iterate over kay/value tiles
    int num_kv_tiles = (T + FA_BK - 1) / FA_BK;

    for (int kv_blk = 0; kv_blk < num_kv_tiles; kv_blk++) {
        int kv_start = kv_blk * FA_BK;
        int kv_idx = kv_start + ty;   // reuse ty for loading KV tiles

        // Load K and V tiles
        if (kv_idx < T && tx < HS) {
            sK[ty * HS + tx] = K[bh_offset + kv_idx * HS + tx];
            sV[ty * HS + tx] = V[bh_offset + kv_idx * HS + tx];
        } else {
            sK[ty * HS + tx] = 0.0f;
            sV[ty * HS + tx] = 0.0f;
        }
        __syncthreads();

        if (q_idx >= T) { __syncthreads(); continue; }

        // Compute attention scores for this tile
        // S[ty, j] = Q[q_idx] · K[kv_start+j]  for j in 0..FA_BK-1
        float m_new = m_i;
        float scores[FA_BK];

        for (int j = 0; j < FA_BK; j++) {
            int kv_global = kv_start + j;
            float s = 0.0f;
            if (kv_global < T && kv_global <= q_idx) {  // causal
                for (int d = 0; d < HS; d++) {
                    s += sQ[ty * HS + d] * sK[j * HS + d];
                }
                s *= scale;
            } else {
                s = -FLT_MAX;
            }
            scores[j] = s;
            m_new = fmaxf(m_new, s);
        }

        // Online softmax update
        float l_new = expf(m_i - m_new) * l_i;
        for (int j = 0; j < FA_BK; j++) {
            l_new += expf(scores[j] - m_new);
        }

        // Rescale running output and accumulate
        o_i = o_i * expf(m_i - m_new);
        for (int j = 0; j < FA_BK; j++) {
            int kv_global = kv_start + j;
            if (kv_global < T && tx < HS) {
                o_i += expf(scores[j] - m_new) * sV[j * HS + tx];
            }
        }

        m_i = m_new;
        l_i = l_new;

        __syncthreads();
    }

    // Write output
    if (q_idx < T && tx < HS) {
        out[bh_offset + q_idx * HS + tx] = o_i / l_i;
    }
}

// =================================================
// Host launchers
// =================================================

// Standard attention (simpler, use for seq_len <= 512)
void launch_attention(
    const float* Q, const float* K, const float* V,
    float* out,
    int B, int H, int T, int HS,
    cudaStream_t stream = 0
) {
    float scale = 1.0f / sqrtf((float)HS);
    dim3 grid(B * H, T);
    dim3 block(T);
    size_t smem = T * sizeof(float);
    attention_forward_kernel<<<grid, block, smem, stream>>>(
        Q, K, V, out, B, H, T, HS, scale
    );
}

// Flash Attention
void launch_flash_attention(
    const float* Q, const float* K, const float* V,
    float* out,
    int B, int H, int T, int HS,
    cudaStream_t stream = 0
) {
    float scale = 1.0f / sqrtf((float)HS);
    dim3 grid(B * H, (T + FA_BQ - 1) / FA_BQ);
    dim3 block(FA_BQ, HS);
    size_t smem = (FA_BQ + 2 * FA_BK) * HS * sizeof(float);
    flash_attention_kernel<<<grid, block, smem, stream>>>(
        Q, K, V, out, B, H, T, HS, scale
    );
}