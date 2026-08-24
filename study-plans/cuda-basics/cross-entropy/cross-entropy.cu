#include <cuda_runtime.h>
#include <float.h>

__global__ void cross_entropy_kernel(const float* logits, const int* target, float* loss, int B, int C) {
    int row = blockIdx.x;
    if (row >= B) return;

    const float* row_logits = logits + (size_t)row * C;
    int tid = threadIdx.x;

    __shared__ float sdata[256];

   
    float local_max = -FLT_MAX;
    for (int j = tid; j < C; j += blockDim.x) {
        local_max = fmaxf(local_max, row_logits[j]);
    }
    sdata[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    __shared__ float row_max;
    if (tid == 0) {
        row_max = sdata[0];
    }
    __syncthreads();

    
    float local_sum = 0.0f;
    for (int j = tid; j < C; j += blockDim.x) {
        local_sum += expf(row_logits[j] - row_max);
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    __shared__ float lse;
    if (tid == 0) {
        lse = logf(sdata[0]);
    }
    __syncthreads();

   
    if (tid == 0) {
        int t = target[row];
        float z_t = row_logits[t];
        float row_loss = -(z_t - row_max - lse);
        atomicAdd(loss, row_loss);
    }
}

__global__ void finalize_kernel(float* loss, int B) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        loss[0] = loss[0] / (float)B;
    }
}


extern "C" void solve(const float* logits, const int* target, float* loss, int B, int C) {
    cudaMemset(loss, 0, sizeof(float));

    int threadsPerBlock = 256;
    int blocksPerGrid = B;

    cross_entropy_kernel<<<blocksPerGrid, threadsPerBlock>>>(logits, target, loss, B, C);
    finalize_kernel<<<1, 1>>>(loss, B);

    cudaDeviceSynchronize();
}