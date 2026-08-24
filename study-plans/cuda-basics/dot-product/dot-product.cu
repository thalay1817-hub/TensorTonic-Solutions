#include <cuda_runtime.h>

__global__ void dot_kernel(const float* A, const float* B, float* result, int N) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float partial = 0.0f;
    for (int i = idx; i < N; i += stride) {
        partial += A[i] * B[i];
    }

    sdata[tid] = partial;
    __syncthreads();

    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}


extern "C" void solve(const float* A, const float* B, float* result, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid < 1) blocksPerGrid = 1;

    
    cudaMemset(result, 0, sizeof(float));

    dot_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, result, N);
    cudaDeviceSynchronize();
}