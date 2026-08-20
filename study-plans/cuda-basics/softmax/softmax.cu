#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

// Atomic max for floats (works for all finite float values, +inf/-inf too)
__device__ float atomicMaxFloat(float* addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) >= value) break;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(value));
    } while (assumed != old);
    return __int_as_float(old);
}

// Pass 1: block-level reduction to find the global maximum
__global__ void maxReduceKernel(const float* input, int N, float* globalMax) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float val = -FLT_MAX;
    if (idx < N) val = input[idx];
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMaxFloat(globalMax, sdata[0]);
    }
}

// Pass 2: block-level reduction to sum exp(x - max)
__global__ void sumExpKernel(const float* input, int N, const float* globalMax, float* globalSum) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float m = *globalMax;
    float val = 0.0f;
    if (idx < N) val = expf(input[idx] - m);
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(globalSum, sdata[0]);
    }
}

// Pass 3: compute final normalized output
__global__ void normalizeKernel(const float* input, float* output, int N,
                                 const float* globalMax, const float* globalSum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float m = *globalMax;
        float s = *globalSum;
        output[idx] = expf(input[idx] - m) / s;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    float* d_max;
    float* d_sum;
    cudaMalloc(&d_max, sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));

    float initMax = -FLT_MAX;
    float initSum = 0.0f;
    cudaMemcpy(d_max, &initMax, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, &initSum, sizeof(float), cudaMemcpyHostToDevice);

    // Step 1: find global max
    maxReduceKernel<<<blocksPerGrid, threadsPerBlock>>>(input, N, d_max);

    // Step 2: sum of exp(x - max)
    sumExpKernel<<<blocksPerGrid, threadsPerBlock>>>(input, N, d_max, d_sum);

    // Step 3: normalize
    normalizeKernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, d_max, d_sum);

    cudaDeviceSynchronize();

    cudaFree(d_max);
    cudaFree(d_sum);
}