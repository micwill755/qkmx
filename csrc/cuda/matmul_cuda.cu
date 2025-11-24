#include "../tensor.h"
#include "cuda_utils.h"

// Simple matmul kernel - each thread computes one output element
__global__ void matmul_kernel(const float* A, const float* B, float* C,
                              int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

Tensor* matmul_cuda(const Tensor* a, const Tensor* b) {
    if (a->dtype != b->dtype) return NULL;
    if (a->ndim < 2 || b->ndim < 2) return NULL;
    
    int M = a->shape[a->ndim - 2];
    int K = a->shape[a->ndim - 1];
    int K_b = b->shape[b->ndim - 2];
    int N = b->shape[b->ndim - 1];
    
    if (K != K_b) return NULL;
    
    // For simplicity, handle only 2D matrices
    // Production code would handle batched matmul
    if (a->ndim != 2 || b->ndim != 2) {
        fprintf(stderr, "CUDA matmul: only 2D matrices supported for now\n");
        return NULL;
    }
    
    int out_shape[2] = {M, N};
    Tensor* out = tensor_create(out_shape, 2, a->dtype);
    if (!out) return NULL;
    out->device = DEVICE_CUDA;
    
    size_t a_bytes = M * K * sizeof(float);
    size_t b_bytes = K * N * sizeof(float);
    size_t c_bytes = M * N * sizeof(float);
    
    // Allocate GPU memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_B, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_C, c_bytes));
    
    // Copy input matrices to GPU
    CUDA_CHECK(cudaMemcpy(d_A, a->data, a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, b->data, b_bytes, cudaMemcpyHostToDevice));
    
    // Launch kernel with 2D grid
    dim3 threads(16, 16);  // 16x16 = 256 threads per block
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    
    matmul_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(out->data, d_C, c_bytes, cudaMemcpyDeviceToHost));
    
    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return out;
}
