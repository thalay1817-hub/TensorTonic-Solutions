#include <cuda_runtime.h>
#include <float.h>

__device__ void combine(float &val_a, int &idx_a, float val_b, int idx_b) {
    if (val_b > val_a || (val_b == val_a && idx_b < idx_a)) {
        val_a = val_b;
        idx_a = idx_b;
    }
}

__global__ void argmax_block_kernel(const float* input, float* block_vals, int* block_idxs, int N) {
    __shared__ float sval[256];
    __shared__ int sidx[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float best_val = -FLT_MAX;
    int best_idx = 0;

    for (int i = idx; i < N; i += stride) {
        combine(best_val, best_idx, input[i], i);
    }

    sval[tid] = best_val;
    sidx[tid] = best_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            combine(sval[tid], sidx[tid], sval[tid + s], sidx[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_vals[blockIdx.x] = sval[0];
        block_idxs[blockIdx.x] = sidx[0];
    }
}

__global__ void argmax_final_kernel(const float* block_vals, const int* block_idxs, int* result, int numBlocks) {
    
    __shared__ float sval[256];
    __shared__ int sidx[256];

    int tid = threadIdx.x;

    float best_val = -FLT_MAX;
    int best_idx = 0;

    for (int i = tid; i < numBlocks; i += blockDim.x) {
        combine(best_val, best_idx, block_vals[i], block_idxs[i]);
    }

    sval[tid] = best_val;
    sidx[tid] = best_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            combine(sval[tid], sidx[tid], sval[tid + s], sidx[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[0] = sidx[0];
    }
}


extern "C" void solve(const float* input, float* result, int N) {
   
    int* result_int = reinterpret_cast<int*>(result);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid < 1) blocksPerGrid = 1;

    float* d_block_vals;
    int* d_block_idxs;
    cudaMalloc(&d_block_vals, blocksPerGrid * sizeof(float));
    cudaMalloc(&d_block_idxs, blocksPerGrid * sizeof(int));

    argmax_block_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, d_block_vals, d_block_idxs, N);

    int finalThreads = 256;
    argmax_final_kernel<<<1, finalThreads>>>(d_block_vals, d_block_idxs, result_int, blocksPerGrid);

    cudaDeviceSynchronize();

    cudaFree(d_block_vals);
    cudaFree(d_block_idxs);
}