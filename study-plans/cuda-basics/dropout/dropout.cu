#include <cuda_runtime.h>

__global__ void dropout_kernel(const float* input, const float* mask, float* output, int N, float p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float scale = 1.0f / (1.0f - p);
        output[idx] = input[idx] * mask[idx] * scale;
    }
}

extern "C" void solve(const float* input, const float* mask, float* output, int N, float p) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid < 1) blocksPerGrid = 1;

    dropout_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, mask, output, N, p);
    cudaDeviceSynchronize();
}