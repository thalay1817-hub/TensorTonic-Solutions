#include <cuda_runtime.h>
#include <math.h>

#define SQRT1_2 0.70710678118654752440f  // 1/sqrt(2)

__global__ void geluKernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = input[idx];
        output[idx] = 0.5f * x * (1.0f + erff(x * SQRT1_2));
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    geluKernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}