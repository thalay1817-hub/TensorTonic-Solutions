#include <cuda_runtime.h>

__global__ void matvec_kernel(const float* A, const float* x, float* y, int M, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            sum += A[i * N + j] * x[j];
        }
        y[i] = sum;
    }
}


extern "C" void solve(const float* A, const float* x, float* y, int M, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;
    matvec_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, x, y, M, N);
    cudaDeviceSynchronize();
}