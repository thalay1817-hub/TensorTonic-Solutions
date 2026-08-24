#include <cuda_runtime.h>
#include <float.h>


__device__ float atomicMaxFloat(float* addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int;
    int assumed;

    while (value > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(value));
        if (assumed == old) break;
    }
    return __int_as_float(old);
}

__global__ void max_kernel(const float* input, float* result, int N) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float partial = -FLT_MAX;
    for (int i = idx; i < N; i += stride) {
        partial = fmaxf(partial, input[i]);
    }

    sdata[tid] = partial;
    __syncthreads();

  
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    
    if (tid == 0) {
        atomicMaxFloat(result, sdata[0]);
    }
}

__global__ void init_result_kernel(float* result) {
    result[0] = -FLT_MAX;
}


extern "C" void solve(const float* input, float* result, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid < 1) blocksPerGrid = 1;

   
    init_result_kernel<<<1, 1>>>(result);

    max_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, result, N);
    cudaDeviceSynchronize();
}