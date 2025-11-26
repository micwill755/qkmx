#include "../tensor.h"
#include "cuda_utils.h"

__global__ void layer_norm_kernel(const float* input, float* output,
                                   const float* gamma, const float* beta,
                                   int outer_size, int D, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < outer_size) {
        const float* row = input + idx * D;
        float* out_row = output + idx * D;
        
        // Compute mean
        float mean = 0.0f;
        for (int j = 0; j < D; j++) {
            mean += row[j];
        }
        mean /= D;
        
        // Compute variance
        float var = 0.0f;
        for (int j = 0; j < D; j++) {
            float diff = row[j] - mean;
            var += diff * diff;
        }
        var /= D;
        
        // Normalize
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int j = 0; j < D; j++) {
            out_row[j] = (row[j] - mean) * inv_std * gamma[j] + beta[j];
        }
    }
}

Tensor* layer_norm_cuda(const Tensor* x, const Tensor* gamma, 
                        const Tensor* beta, float eps) {
    Tensor* out = tensor_create(x->shape, x->ndim, x->dtype);
    if (!out) return NULL;
    out->device = DEVICE_CUDA;
    
    int D = x->shape[x->ndim - 1];
    int outer_size = x->size / D;
    
    size_t x_bytes = x->size * sizeof(float);
    size_t param_bytes = D * sizeof(float);
    
    // Allocate GPU memory
    float *d_input, *d_output, *d_gamma, *d_beta;
    CUDA_CHECK(cudaMalloc(&d_input, x_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, x_bytes));
    CUDA_CHECK(cudaMalloc(&d_gamma, param_bytes));
    CUDA_CHECK(cudaMalloc(&d_beta, param_bytes));
    
    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_input, x->data, x_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, gamma->data, param_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, beta->data, param_bytes, cudaMemcpyHostToDevice));
    
    // Launch kernel - one thread per row
    int threads = 256;
    int blocks = (outer_size + threads - 1) / threads;
    layer_norm_kernel<<<blocks, threads>>>(d_input, d_output, d_gamma, d_beta,
                                           outer_size, D, eps);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(out->data, d_output, x_bytes, cudaMemcpyDeviceToHost));
    
    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    
    return out;
}
