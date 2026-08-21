#include <cuda_runtime.h>

__global__ void layerNormKernel(const float* input, const float* gamma, const float* beta,
                                 float* output, int M, int N, float eps) {
    const int BLOCK_SIZE = 256;

    int row = blockIdx.x;
    if (row >= M) return;

    const float* rowInput = input + (size_t)row * N;
    float* rowOutput = output + (size_t)row * N;

    __shared__ float sSum[256];
    __shared__ float sSumSq[256];
    __shared__ float sMean;
    __shared__ float sInvStd;

    int tid = threadIdx.x;

    
    float localSum = 0.0f;
    float localSumSq = 0.0f;
    for (int j = tid; j < N; j += blockDim.x) {
        float val = rowInput[j];
        localSum += val;
        localSumSq += val * val;
    }

    sSum[tid] = localSum;
    sSumSq[tid] = localSumSq;
    __syncthreads();

    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sSum[tid] += sSum[tid + stride];
            sSumSq[tid] += sSumSq[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mean = sSum[0] / N;
        float meanSq = sSumSq[0] / N;
        float var = meanSq - mean * mean;
        sMean = mean;
        sInvStd = rsqrtf(var + eps);
    }
    __syncthreads();

    float mean = sMean;
    float invStd = sInvStd;

    for (int j = tid; j < N; j += blockDim.x) {
        float val = rowInput[j];
        rowOutput[j] = (val - mean) * invStd * gamma[j] + beta[j];
    }
}


extern "C" void solve(const float* input, const float* gamma, const float* beta,
                       float* output, int M, int N, float eps) {
    int threads = 256;
    if (threads > N) {
        
        int t = 1;
        while (t < N) t <<= 1;
        threads = t;
    }
    dim3 blockDim(threads);
    dim3 gridDim(M);

    layerNormKernel<<<gridDim, blockDim>>>(input, gamma, beta, output, M, N, eps);
    cudaDeviceSynchronize();
}