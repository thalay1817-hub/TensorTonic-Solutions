#include <cuda_runtime.h>

__global__ void sq_sum_kernel(const float* input, float* sum, int N) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float partial = 0.0f;
    for (int i = idx; i < N; i += stride) {
        float v = input[i];
        partial += v * v;
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
        atomicAdd(sum, sdata[0]);
    }
}

__global__ void divide_kernel(const float* input, float* output, const float* sum, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float norm = sqrtf(*sum);
    for (int i = idx; i < N; i += stride) {
        output[i] = input[i] / norm;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid < 1) blocksPerGrid = 1;

    float* d_sum;
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float));

    sq_sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, d_sum, N);
    divide_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, d_sum, N);

    cudaDeviceSynchronize();

    cudaFree(d_sum);
}