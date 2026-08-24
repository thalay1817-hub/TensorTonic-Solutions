#include <cuda_runtime.h>

__global__ void rmsnorm_kernel(const float* input, const float* gamma, float* output,
                                int M, int N, float eps) {
    int row = blockIdx.x;
    if (row >= M) return;

    const float* in_row = input + (size_t)row * N;
    float* out_row = output + (size_t)row * N;

    __shared__ float sdata[256];

    int tid = threadIdx.x;

    
    float partial = 0.0f;
    for (int j = tid; j < N; j += blockDim.x) {
        float v = in_row[j];
        partial += v * v;
    }
    sdata[tid] = partial;
    __syncthreads();

    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    __shared__ float inv_rms;
    if (tid == 0) {
        float meanSquare = sdata[0] / (float)N;
        inv_rms = rsqrtf(meanSquare + eps);
    }
    __syncthreads();

    
    for (int j = tid; j < N; j += blockDim.x) {
        out_row[j] = in_row[j] * inv_rms * gamma[j];
    }
}


extern "C" void solve(const float* input, const float* gamma, float* output,
                       int M, int N, float eps) {
    int threadsPerBlock = 256;
    int blocksPerGrid = M;

    rmsnorm_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, gamma, output, M, N, eps);
    cudaDeviceSynchronize();
}