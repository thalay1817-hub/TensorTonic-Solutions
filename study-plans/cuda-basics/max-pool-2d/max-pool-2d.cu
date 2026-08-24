#include <cuda_runtime.h>
#include <float.h>

__global__ void maxpool2d_kernel(const float* input, float* output,
                                  int H, int W, int kH, int kW, int sH, int sW,
                                  int outH, int outW) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; 
    int i = blockIdx.y * blockDim.y + threadIdx.y; 

    if (i < outH && j < outW) {
        float best = -FLT_MAX;
        int base_row = i * sH;
        int base_col = j * sW;

        for (int a = 0; a < kH; ++a) {
            for (int b = 0; b < kW; ++b) {
                float v = input[(base_row + a) * W + (base_col + b)];
                best = fmaxf(best, v);
            }
        }

        output[i * outW + j] = best;
    }
}


extern "C" void solve(const float* input, float* output,
                       int H, int W, int kH, int kW, int sH, int sW) {
    int outH = (H - kH) / sH + 1;
    int outW = (W - kW) / sW + 1;

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((outW + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (outH + threadsPerBlock.y - 1) / threadsPerBlock.y);

    maxpool2d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, H, W, kH, kW, sH, sW, outH, outW);
    cudaDeviceSynchronize();
}