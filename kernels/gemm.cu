// kernels/gemm.cu
// Tiled matrix multiply: C = A @ B
// A: (M, K), B: (K, N), C: (M, N)
//
// Grid:  (ceil(N/BN), ceil(M/BM))
// Block: (BN, BM)
// Shared: BM*BK + BK*BN floats

#include <cuda_runtime.h>
#include <stdio.h>

#define BM 32
#define BN 32
#define BK 32

// ===================================================================
// Naive baseline (for correctness reference only — not used in engine)
// ===================================================================

__global__ void gemm_naive(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}

// ===================================================================
// Tiled GEMM — shared memory reduces global memory traffic by BK factor
// ===================================================================

__global__ void gemm_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row = blockIdx.y * BM + ty;
    int col = blockIdx.x * BN + tx;

    float acc = 0.0f;

    for (int tile = 0; tile < (K + BK - 1) / BK; tile++) {
        // Load tile of A into shared memory
        int a_col = tile * BK + tx;
        sA[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        // Load tile of B into shared memory
        int b_row = tile * BK + ty;
        sB[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        // Accumulate dot product over tile
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            acc += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// ===================================================================
// Batched GEMM — runs the same (M,K)@(K,N) for each item in batch
// ===================================================================

__global__ void gemm_batched(
    const float* __restrict__ A,   // (B, M, K)
    const float* __restrict__ B,   // (B, K, N)
    float* __restrict__ C,   // (B, M, N)
    int M, int N, int K
) {
    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];

    int batch = blockIdx.z;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row = blockIdx.y * BM + ty;
    int col = blockIdx.x * BN + tx;

    // Offset pointers to the correct batch
    const float* Ab = A + batch * M * K;
    const float* Bb = B + batch * K * N;
    float*       Cb = C + batch * M * N;

    float acc = 0.0f;

    for (int tile = 0; tile < (K + BK - 1) / BK; tile++) {
        int a_col = tile * BK + tx;
        sA[ty][tx] = (row < M && a_col < K) ? Ab[row * K + a_col] : 0.0f;

        int b_row = tile * BK + ty;
        sB[ty][tx] = (b_row < K && col < N) ? Bb[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            acc += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        Cb[row * N + col] = acc;
    }
}

// ===================================================================
// Host launcher functions (called from engine/model.cpp or tests)
// ===================================================================

void launch_gemm(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    cudaStream_t stream = 0
) {
    dim3 block(BN, BM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_tiled<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

void launch_gemm_batched(
    const float* A, const float* B, float* C,
    int batch, int M, int N, int K,
    cudaStream_t stream = 0
) {
    dim3 block(BN, BM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, batch);
    gemm_batched<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}