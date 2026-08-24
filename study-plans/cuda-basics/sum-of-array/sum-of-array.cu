#include <cuda_runtime.h>

__global__ void sum_kernel(const float* input, float* result, int N) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float partial = 0.0f;
    for (int i = idx; i < N; i += stride) {
        partial += input[i];
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


extern "C" void solve(const float* input, float* result, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid < 1) blocksPerGrid = 1;

   
    cudaMemset(result, 0, sizeof(float));

    sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, result, N);
    cudaDeviceSynchronize();
}