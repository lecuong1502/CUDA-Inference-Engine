// kernels/activation.cu
// Softmax: numerically stable (online max trick)
// Input shape: (rows, cols) — applies softmax over last dimension
//
// Grid:  (rows)
// Block: (min(cols, 1024))

#include <cuda_runtime.h>
#include <float.h>

// ===================================================================
// Softmax — one block per row, two-pass: max then exp/sum
// Works for cols up to 1024 (one block). See softmax_large for wider.
// ===================================================================

__global__ void softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows, int cols
) {
    extern __shared__ float smem[];    // cols floats

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) return;

    const float* in_row  = input  + row * cols;
    float* out_row = output + row * cols;

    // Pass 1: Load + Find row max
    float val = (tid < cols) ? in_row[tid] : -FLT_MAX;
    smem[tid] = val;
    __syncthreads();

    // Parallel reduction for max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }
    float row_max = smem[0];
    __syncthreads();

    // Pass 2: exp(x - max) + sum
    float exp_val = (tid < cols) ? expf(val - row_max) : 0.0f;
    smem[tid] = exp_val;
    __syncthreads();

    // Parallel reduction for sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    float row_sum = smem[0];

    // Write output
    if (tid < cols) {
        out_row[tid] = exp_val / row_sum;
    }
}

// ===================================================================
// Softmax for attention scores: input (batch, heads, seq, seq)
// Flattened to (batch*heads*seq, seq) before calling softmax_kernel
// with an optional causal mask applied before exp
// ===================================================================

__global__ void softmax_causal_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int seq_len                // cols = seq_len (square attn)
) {
    extern __shared__ float smem[];

    int row = blockIdx.x;   // which query position (0..seq_len-1)
    int tid = threadIdx.x;

    const float* in_row  = input  + row * seq_len;
    float* out_row = output + row * seq_len;

    // Mask: position tid is valid only if tid <= row (casual)
    float val = (tid <= row) ? in_row[tid] : -FLT_MAX;

    // Same two-pass softmax
    smem[tid] = val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        __syncthreads();
    }
    float row_max = smem[0];
    __syncthreads();

    float exp_val = (tid <= row) ? expf(val - row_max) : 0.0f;
    smem[tid] = exp_val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    float row_sum = smem[0];

    if (tid < seq_len) {
        out_row[tid] = (tid <= row) ? (exp_val / row_sum) : 0.0f;
    }
}

// ===================================================================
// GELU activation — used in FFN
// Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
// ===================================================================

__device__ __forceinline__ float gelu(float x) {
    const float c = 0.7978845608f;  // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}

__global__ void gelu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        output[idx] = gelu(input[idx]);
    }
}

// ===================================================================
// Host launchers
// ===================================================================

void launch_softmax(
    const float* input, float* output,
    int rows, int cols,
    cudaStream_t stream = 0
) {
    int threads = 1;
    while (threads < cols) threads <<= 1;   // next power of 2 >= cols
    threads = min(threads, 1024);

    size_t smem = threads + sizeof(float);
    softmax_kernel<<<rows, threads, smem, stream>>>(input, output, rows, cols);
}

void launch_softmax_causal(
    const float* input, float* output,
    int batch_heads, int seq_len,
    cudaStream_t stream = 0
) {
    int threads = 1;
    while (threads < seq_len) threads <<= 1;
    threads = min(threads, 1024);

    size_t smem = threads * sizeof(float);
    // Launch one block per (batch*head, query_position) pair
    softmax_causal_kernel<<<batch_heads * seq_len, threads, smem, stream>>>(
        input, output, seq_len
    );
}

void launch_gelu(
    const float* input, float* output,
    int n,
    cudaStream_t stream = 0
) {
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    gelu_kernel<<<blocks, threads, 0, stream>>>(input, output, n);
}