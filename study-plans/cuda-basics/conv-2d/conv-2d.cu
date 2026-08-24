#include <cuda_runtime.h>

__global__ void conv2d_kernel(const float* input, const float* kernel, float* output,
                               int H, int W, int kH, int kW) {
    int outH = H - kH + 1;
    int outW = W - kW + 1;

    int j = blockIdx.x * blockDim.x + threadIdx.x; 
    int i = blockIdx.y * blockDim.y + threadIdx.y; 

    if (i < outH && j < outW) {
        float sum = 0.0f;
        for (int a = 0; a < kH; ++a) {
            for (int b = 0; b < kW; ++b) {
                sum += input[(i + a) * W + (j + b)] * kernel[a * kW + b];
            }
        }
        output[i * outW + j] = sum;
    }
}


extern "C" void solve(const float* input, const float* kernel, float* output,
                       int H, int W, int kH, int kW) {
    int outH = H - kH + 1;
    int outW = W - kW + 1;

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((outW + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (outH + threadsPerBlock.y - 1) / threadsPerBlock.y);

    conv2d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, H, W, kH, kW);
    cudaDeviceSynchronize();
}