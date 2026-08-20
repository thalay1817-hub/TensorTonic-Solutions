#include <cuda_runtime.h>

__global__ void leakyReluKernel(const float* input, float* output, int N, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = input[idx];
        output[idx] = fmaxf(x, alpha * x);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, float alpha) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    leakyReluKernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, alpha);
    cudaDeviceSynchronize();
}