#include <cuda_runtime.h>
#include <math.h>

constexpr int BLOCK_SIZE = 256;


__global__ void reduce_kernel(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ dotAcc,
                               float* __restrict__ normAAcc,
                               float* __restrict__ normBAcc,
                               int N)
{
    __shared__ float sdot[BLOCK_SIZE];
    __shared__ float sA[BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float dot = 0.0f, sa = 0.0f, sb = 0.0f;
    for (int i = idx; i < N; i += stride) {
        float a = A[i];
        float b = B[i];
        dot += a * b;
        sa  += a * a;
        sb  += b * b;
    }

    sdot[tid] = dot;
    sA[tid] = sa;
    sB[tid] = sb;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdot[tid] += sdot[tid + s];
            sA[tid]   += sA[tid + s];
            sB[tid]   += sB[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(dotAcc, sdot[0]);
        atomicAdd(normAAcc, sA[0]);
        atomicAdd(normBAcc, sB[0]);
    }
}


__global__ void finalize_kernel(const float* __restrict__ dotAcc,
                                 const float* __restrict__ normAAcc,
                                 const float* __restrict__ normBAcc,
                                 float* __restrict__ result)
{
    float dot = *dotAcc;
    float normA = sqrtf(*normAAcc);
    float normB = sqrtf(*normBAcc);
    result[0] = dot / (normA * normB);
}


extern "C" void solve(const float* A, const float* B, float* result, int N)
{
    float* d_dot = nullptr;
    float* d_normA = nullptr;
    float* d_normB = nullptr;

    cudaMalloc(&d_dot, sizeof(float));
    cudaMalloc(&d_normA, sizeof(float));
    cudaMalloc(&d_normB, sizeof(float));

    cudaMemset(d_dot, 0, sizeof(float));
    cudaMemset(d_normA, 0, sizeof(float));
    cudaMemset(d_normB, 0, sizeof(float));

    int threads = BLOCK_SIZE;
    int blocks = (N + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;   // grid-stride loop handles the rest
    if (blocks < 1) blocks = 1;

    reduce_kernel<<<blocks, threads>>>(A, B, d_dot, d_normA, d_normB, N);
    finalize_kernel<<<1, 1>>>(d_dot, d_normA, d_normB, result);

    cudaFree(d_dot);
    cudaFree(d_normA);
    cudaFree(d_normB);

    cudaDeviceSynchronize();
}