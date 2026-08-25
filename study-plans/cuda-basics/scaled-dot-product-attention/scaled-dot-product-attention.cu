#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

__global__ void scores_kernel(const float* __restrict__ Q,
                               const float* __restrict__ K,
                               float* __restrict__ S,
                               int N, int D)
{
    int i = blockIdx.x;   // query row
    int j = threadIdx.x;  // key row

    if (i < N && j < N) {
        float sum = 0.0f;
        const float* qrow = Q + (size_t)i * D;
        const float* krow = K + (size_t)j * D;
        for (int d = 0; d < D; ++d) {
            sum += qrow[d] * krow[d];
        }
        S[(size_t)i * N + j] = sum * rsqrtf((float)D);
    }
}


__global__ void softmax_kernel(float* __restrict__ S, int N)
{
    extern __shared__ float sdata[];

    int i = blockIdx.x;
    int j = threadIdx.x;
    int threads = blockDim.x;

    float val = (j < N) ? S[(size_t)i * N + j] : -FLT_MAX;
    sdata[j] = val;
    __syncthreads();

    // reduction: max
    for (int stride = threads / 2; stride > 0; stride >>= 1) {
        if (j < stride) sdata[j] = fmaxf(sdata[j], sdata[j + stride]);
        __syncthreads();
    }
    float maxval = sdata[0];
    __syncthreads();

    // exponentiate
    float expval = (j < N) ? expf(val - maxval) : 0.0f;
    sdata[j] = expval;
    __syncthreads();

    // reduction: sum
    for (int stride = threads / 2; stride > 0; stride >>= 1) {
        if (j < stride) sdata[j] += sdata[j + stride];
        __syncthreads();
    }
    float sumval = sdata[0];
    __syncthreads();

    if (j < N) {
        S[(size_t)i * N + j] = expval / sumval;
    }
}

__global__ void aggregate_kernel(const float* __restrict__ S,
                                  const float* __restrict__ V,
                                  float* __restrict__ output,
                                  int N, int D)
{
    int i = blockIdx.x;   // query row
    int d = threadIdx.x;  // feature

    if (i < N && d < D) {
        float sum = 0.0f;
        const float* srow = S + (size_t)i * N;
        for (int j = 0; j < N; ++j) {
            sum += srow[j] * V[(size_t)j * D + d];
        }
        output[(size_t)i * D + d] = sum;
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output,
                       int N, int D)
{
    float* d_S = nullptr;
    cudaMalloc(&d_S, (size_t)N * N * sizeof(float));

    // Stage 1: scaled scores
    scores_kernel<<<N, N>>>(Q, K, d_S, N, D);

    // Stage 2: row-wise softmax
    int threads = 1;
    while (threads < N) threads <<= 1;   // next power of two >= N (N <= 1024)
    softmax_kernel<<<N, threads, threads * sizeof(float)>>>(d_S, N);

    // Stage 3: weighted aggregation with V
    aggregate_kernel<<<N, D>>>(d_S, V, output, N, D);

    cudaFree(d_S);
    cudaDeviceSynchronize();
}