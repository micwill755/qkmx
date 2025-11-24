#include "../tensor.h"
#include "cuda_utils.h"

__global__ void gelu_kernel(const float* input, float* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Get thread ID
    
    if (idx < size) {  // Bounds check
        float x = input[idx];
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        output[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

// Host function - this is what gets called from dispatch
Tensor* gelu_cuda(const Tensor* t) {
    // Create output tensor
    Tensor* out = tensor_create(t->shape, t->ndim, t->dtype);
    if (!out) return NULL;
    out->device = DEVICE_CUDA;
    
    size_t bytes = t->size * sizeof(float);
    
    // Step 1: Allocate GPU memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    
    // Step 2: Copy input data from CPU to GPU
    CUDA_CHECK(cudaMemcpy(d_input, t->data, bytes, cudaMemcpyHostToDevice));
    
    // Step 3: Configure kernel launch parameters
    int threads = 256;  // 256 threads per block (common choice)
    int blocks = (t->size + threads - 1) / threads;  // Enough blocks to cover all elements
    
    // Step 4: Launch the kernel
    gelu_kernel<<<blocks, threads>>>(d_input, d_output, t->size);
    
    // Step 5: Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Step 6: Wait for kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Step 7: Copy result from GPU back to CPU
    CUDA_CHECK(cudaMemcpy(out->data, d_output, bytes, cudaMemcpyDeviceToHost));
    
    // Step 8: Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);
    
    return out;
}

// Simple softmax kernel - one thread per row
__global__ void softmax_kernel(const float* input, float* output, 
                               int outer_size, int dim_size, int inner_size) {
    // Each thread handles one "row" (sequence of elements to normalize)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < outer_size * inner_size) {
        int outer = idx / inner_size;
        int inner = idx % inner_size;
        int base_idx = outer * dim_size * inner_size + inner;
        
        // Step 1: Find max for numerical stability
        float max_val = input[base_idx];
        for (int d = 1; d < dim_size; d++) {
            int offset = base_idx + d * inner_size;
            if (input[offset] > max_val) {
                max_val = input[offset];
            }
        }
        
        // Step 2: Compute exp and sum
        float sum = 0.0f;
        for (int d = 0; d < dim_size; d++) {
            int offset = base_idx + d * inner_size;
            float exp_val = expf(input[offset] - max_val);
            output[offset] = exp_val;
            sum += exp_val;
        }
        
        // Step 3: Normalize
        for (int d = 0; d < dim_size; d++) {
            int offset = base_idx + d * inner_size;
            output[offset] /= sum;
        }
    }
}

Tensor* softmax_cuda(const Tensor* t, int dim) {
    Tensor* out = tensor_create(t->shape, t->ndim, t->dtype);
    if (!out) return NULL;
    out->device = DEVICE_CUDA;
    
    // Handle negative dim
    if (dim < 0) dim += t->ndim;
    
    // Calculate dimensions
    int dim_size = t->shape[dim];
    int outer_size = 1;
    for (int i = 0; i < dim; i++) outer_size *= t->shape[i];
    int inner_size = 1;
    for (int i = dim + 1; i < t->ndim; i++) inner_size *= t->shape[i];
    
    size_t bytes = t->size * sizeof(float);
    
    // Allocate GPU memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    
    // Copy input to GPU
    CUDA_CHECK(cudaMemcpy(d_input, t->data, bytes, cudaMemcpyHostToDevice));
    
    // Launch kernel - one thread per "row"
    int total_rows = outer_size * inner_size;
    int threads = 256;
    int blocks = (total_rows + threads - 1) / threads;
    
    softmax_kernel<<<blocks, threads>>>(d_input, d_output, 
                                        outer_size, dim_size, inner_size);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(out->data, d_output, bytes, cudaMemcpyDeviceToHost));
    
    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);
    
    return out;
}
