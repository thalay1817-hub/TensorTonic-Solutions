#include <cuda_runtime.h>

constexpr int TILE_DIM = 16;


__global__ void tiled_matmul_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int M, int K, int N)
{
    __shared__ float tile_A[TILE_DIM][TILE_DIM];
    __shared__ float tile_B[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_DIM + ty;  // i index into C / A
    int col = blockIdx.x * TILE_DIM + tx;  // j index into C / B

    float acc = 0.0f;

    int numChunks = (K + TILE_DIM - 1) / TILE_DIM;

    for (int chunk = 0; chunk < numChunks; ++chunk) {
        int kBase = chunk * TILE_DIM;

        // Load A tile: element (row, kBase + tx)
        int aCol = kBase + tx;
        if (row < M && aCol < K) {
            tile_A[ty][tx] = A[(size_t)row * K + aCol];
        } else {
            tile_A[ty][tx] = 0.0f;
        }

        // Load B tile: element (kBase + ty, col)
        int bRow = kBase + ty;
        if (bRow < K && col < N) {
            tile_B[ty][tx] = B[(size_t)bRow * N + col];
        } else {
            tile_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma 
        for (int k = 0; k < TILE_DIM; ++k) {
            acc += tile_A[ty][k] * tile_B[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[(size_t)row * N + col] = acc;
    }
}


extern "C" void solve(const float* A, const float* B, float* C,
                       int M, int N, int K)
{
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM,
              (M + TILE_DIM - 1) / TILE_DIM);

    tiled_matmul_kernel<<<grid, block>>>(A, B, C, M, K, N);

    cudaDeviceSynchronize();
}