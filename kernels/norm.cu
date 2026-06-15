// kernels/norm.cu
// Fused layer norm: single-pass mean + variance + normalize + affine
// Input shape: (rows, hidden_dim)
// gamma, beta: (hidden_dim,) — learned affine parameters
//
// Grid:  (rows)
// Block: (min(hidden_dim, 1024))

#include <cuda_runtime.h>

// ===================================================================
// Layer norm — fused single-pass
// Uses Welford's online algorithm for numerically stable mean+variance
// in one pass over the data, then normalizes in-place
// ===================================================================

__global__ void layernorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,   // scale
    const float* __restrict__ beta,    // shift
    float* __restrict__ output,
    int hidden_dim,
    float eps
) {
    extern __shared__ float smem[];     // 2 * blockDim.x floats

    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float* in_row  = input  + row * hidden_dim;
    float* out_row = output + row * hidden_dim;

    // Welford online mean + M2
    float mean = 0.0f, M2 = 0.0f;
    int count = 0;

    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float x = in_row[i];
        count++;
        float delta  = x - mean;
        mean += delta / count;
        float delta2 = x - mean;
        M2 += delta * delta2;
    }

    // Store partial mean and M2 in shared memory
    float* smem_mean = smem;
    float* smem_m2 = smem + blockDim.x;

    smem_mean[tid] = mean * count;   // sum
    smem_m2[tid] = M2 + mean * mean * count;  // sum of squares
    __syncthreads();

    // Reduce: sum across threads
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem_mean[tid] += smem_mean[tid + stride];
            smem_m2[tid]   += smem_m2[tid + stride];
        }
        __syncthreads();
    }

    float row_mean = smem_mean[0] / hidden_dim;
    float row_var = smem_m2[0] / hidden_dim - row_mean * row_mean;
    float inv_std = rsqrtf(row_var + eps);

    // Normalize + Affine
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float norm = (in_row[i] - row_mean) * inv_std;
        out_row[i] = gamma[i] * norm + beta[i];
    }
}

// ===================================================================
// Fused residual add + layer norm
// out = layernorm(residual + input)
// Avoids a separate kernel + global memory round-trip for the add
// ===================================================================

__global__ void add_layernorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ residual,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int hidden_dim,
    float eps
) {
    extern __shared__ float smem[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float* in_row  = input + row * hidden_dim;
    const float* res_row = residual + row * hidden_dim;
    float* out_row = output + row * hidden_dim;

    float* smem_mean = smem;
    float* smem_m2 = smem + blockDim.x;

    // Pass 1: Compute residual sum + sum of squares
    float local_sum = 0.0f, local_sq = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float x = in_row[i] + res_row[i];
        local_sum += x;
        local_sq += x * x;
    }

    smem_mean[tid] = local_sum;
    smem_m2[tid]   = local_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem_mean[tid] += smem_mean[tid + stride];
            smem_m2[tid] += smem_m2[tid + stride];
        }
        __syncthreads();
    }

    float row_mean = smem_mean[0] / hidden_dim;
    float row_var = smem_m2[0] / hidden_dim - row_mean * row_mean;
    float inv_std = rsqrtf(row_var + eps);

    // Pass 2: Normalize + Affine
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float x = in_row[i] + res_row[i];
        float norm = (x - row_mean) * inv_std;
        out_row[i] = gamma[i] * norm + beta[i];
    }
}

// ===================================================================
// Host launchers
// ===================================================================

void launch_layernorm(
    const float* input,
    const float* gamma, const float* beta,
    float* output,
    int rows, int hidden_dim,
    float eps = 1e-5f,
    cudaStream_t stream = 0
) {
    int threads = min(hidden_dim, 1024);
    // Round up to warp size (32)
    threads = ((threads + 31) / 32) * 32;

    size_t smem = 2 * threads * sizeof(float);
    layernorm_kernel<<<rows, threads, smem, stream>>>(
        input, gamma, beta, output, hidden_dim, eps
    );
}

void launch_add_layernorm(
    const float* input, const float* residual,
    const float* gamma, const float* beta,
    float* output,
    int rows, int hidden_dim,
    float eps = 1e-5f,
    cudaStream_t stream = 0
) {
    int threads = min(hidden_dim, 1024);
    threads = ((threads + 31) / 32) * 32;

    size_t smem = 2 * threads * sizeof(float);
    add_layernorm_kernel<<<rows, threads, smem, stream>>>(
        input, residual, gamma, beta, output, hidden_dim, eps
    );
}