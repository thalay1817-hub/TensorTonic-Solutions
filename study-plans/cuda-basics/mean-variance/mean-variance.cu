#include <cuda_runtime.h>

__global__ void sum_sumsq_kernel(const float* input, float* sum, float* sumsq, int N) {
    __shared__ float sdata_sum[256];
    __shared__ float sdata_sumsq[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float partial_sum = 0.0f;
    float partial_sumsq = 0.0f;
    for (int i = idx; i < N; i += stride) {
        float v = input[i];
        partial_sum += v;
        partial_sumsq += v * v;
    }

    sdata_sum[tid] = partial_sum;
    sdata_sumsq[tid] = partial_sumsq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_sum[tid] += sdata_sum[tid + s];
            sdata_sumsq[tid] += sdata_sumsq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sum, sdata_sum[0]);
        atomicAdd(sumsq, sdata_sumsq[0]);
    }
}

__global__ void finalize_kernel(const float* sum, const float* sumsq, float* mean_out, float* var_out, int N) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float mean = *sum / (float)N;
        float meansq = *sumsq / (float)N;
        float var = meansq - mean * mean;
        if (var < 0.0f) var = 0.0f; 
        mean_out[0] = mean;
        var_out[0] = var;
    }
}


extern "C" void solve(const float* input, float* mean_out, float* var_out, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid < 1) blocksPerGrid = 1;

    float* d_sum;
    float* d_sumsq;
    cudaMalloc(&d_sum, sizeof(float));
    cudaMalloc(&d_sumsq, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float));
    cudaMemset(d_sumsq, 0, sizeof(float));

    sum_sumsq_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, d_sum, d_sumsq, N);
    finalize_kernel<<<1, 1>>>(d_sum, d_sumsq, mean_out, var_out, N);

    cudaDeviceSynchronize();

    cudaFree(d_sum);
    cudaFree(d_sumsq);
}