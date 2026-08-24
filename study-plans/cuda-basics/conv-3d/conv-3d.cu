#include <cuda_runtime.h>

__global__ void conv3d_kernel(const float* input, const float* kernel, float* output,
                               int D, int H, int W, int kD, int kH, int kW) {
    int outD = D - kD + 1;
    int outH = H - kH + 1;
    int outW = W - kW + 1;

    int j = blockIdx.x * blockDim.x + threadIdx.x; 
    int i = blockIdx.y * blockDim.y + threadIdx.y; 
    int d = blockIdx.z * blockDim.z + threadIdx.z; 

    if (d < outD && i < outH && j < outW) {
        float sum = 0.0f;
        for (int a = 0; a < kD; ++a) {
            for (int b = 0; b < kH; ++b) {
                for (int c = 0; c < kW; ++c) {
                    float in_val = input[(d + a) * H * W + (i + b) * W + (j + c)];
                    float k_val = kernel[a * kH * kW + b * kW + c];
                    sum += in_val * k_val;
                }
            }
        }
        output[d * outH * outW + i * outW + j] = sum;
    }
}


extern "C" void solve(const float* input, const float* kernel, float* output,
                       int D, int H, int W, int kD, int kH, int kW) {
    int outD = D - kD + 1;
    int outH = H - kH + 1;
    int outW = W - kW + 1;

    dim3 threads(8, 8, 8);
    dim3 blocks((outW + threads.x - 1) / threads.x,
                (outH + threads.y - 1) / threads.y,
                (outD + threads.z - 1) / threads.z);

    conv3d_kernel<<<blocks, threads>>>(input, kernel, output, D, H, W, kD, kH, kW);
    cudaDeviceSynchronize();
}