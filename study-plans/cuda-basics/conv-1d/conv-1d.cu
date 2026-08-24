#include <cuda_runtime.h>

__global__ void conv1d_kernel(const float* input, const float* kernel, float* output, int N, int kN) {
    int outN = N - kN + 1;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < outN) {
        float sum = 0.0f;
        for (int a = 0; a < kN; ++a) {
            sum += input[i + a] * kernel[a];
        }
        output[i] = sum;
    }
}


extern "C" void solve(const float* input, const float* kernel, float* output, int N, int kN) {
    int outN = N - kN + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (outN + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid < 1) blocksPerGrid = 1;

    conv1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, N, kN);
    cudaDeviceSynchronize();
}