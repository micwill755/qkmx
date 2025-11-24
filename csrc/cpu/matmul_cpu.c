#include "../tensor.h"
#include <stdlib.h>

#define MATMUL_IMPL(type, a_data, b_data, out_data, M, K, N) \
    do { \
        for (int i = 0; i < (M); i++) { \
            for (int j = 0; j < (N); j++) { \
                type sum = 0; \
                for (int k = 0; k < (K); k++) { \
                    sum += (a_data)[i * (K) + k] * (b_data)[k * (N) + j]; \
                } \
                (out_data)[i * (N) + j] = sum; \
            } \
        } \
    } while(0)

Tensor* matmul_cpu(const Tensor* a, const Tensor* b) {
    if (a->dtype != b->dtype) return NULL;
    if (a->ndim < 2 || b->ndim < 2) return NULL;
    
    int M = a->shape[a->ndim - 2];
    int K = a->shape[a->ndim - 1];
    int K_b = b->shape[b->ndim - 2];
    int N = b->shape[b->ndim - 1];
    
    if (K != K_b) return NULL;
    
    int out_ndim = (a->ndim > b->ndim) ? a->ndim : b->ndim;
    int batch_ndim = out_ndim - 2;
    
    int* out_shape = malloc(out_ndim * sizeof(int));
    
    for (int i = 0; i < batch_ndim; i++) {
        int idx_a = i - (batch_ndim - (a->ndim - 2));
        int idx_b = i - (batch_ndim - (b->ndim - 2));
        
        int dim_a = (idx_a >= 0) ? a->shape[idx_a] : 1;
        int dim_b = (idx_b >= 0) ? b->shape[idx_b] : 1;
        
        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            free(out_shape);
            return NULL;
        }
        out_shape[i] = (dim_a > dim_b) ? dim_a : dim_b;
    }
    out_shape[out_ndim - 2] = M;
    out_shape[out_ndim - 1] = N;
    
    Tensor* out = tensor_create(out_shape, out_ndim, a->dtype);
    free(out_shape);
    
    size_t total_batch = 1;
    for (int i = 0; i < batch_ndim; i++) {
        total_batch *= out->shape[i];
    }
    
    if (a->dtype == DTYPE_FLOAT32) {
        float* a_data = (float*)a->data;
        float* b_data = (float*)b->data;
        float* out_data = (float*)out->data;
        
        size_t matrix_size_a = M * K;
        size_t matrix_size_b = K * N;
        size_t matrix_size_out = M * N;
        
        size_t* strides_a = malloc(batch_ndim * sizeof(size_t));
        size_t* strides_b = malloc(batch_ndim * sizeof(size_t));
        
        size_t stride_a = matrix_size_a;
        size_t stride_b = matrix_size_b;
        for (int i = batch_ndim - 1; i >= 0; i--) {
            int idx_a = i - (batch_ndim - (a->ndim - 2));
            int idx_b = i - (batch_ndim - (b->ndim - 2));
            
            strides_a[i] = (idx_a >= 0 && a->shape[idx_a] > 1) ? stride_a : 0;
            strides_b[i] = (idx_b >= 0 && b->shape[idx_b] > 1) ? stride_b : 0;
            
            if (idx_a >= 0) stride_a *= a->shape[idx_a];
            if (idx_b >= 0) stride_b *= b->shape[idx_b];
        }
        
        for (size_t batch = 0; batch < total_batch; batch++) {
            size_t idx_a = 0, idx_b = 0;
            size_t temp = batch;
            
            for (int i = batch_ndim - 1; i >= 0; i--) {
                size_t coord = temp % out->shape[i];
                temp /= out->shape[i];
                idx_a += coord * strides_a[i];
                idx_b += coord * strides_b[i];
            }
            
            float* a_mat = a_data + idx_a;
            float* b_mat = b_data + idx_b;
            float* out_mat = out_data + batch * matrix_size_out;
            
            MATMUL_IMPL(float, a_mat, b_mat, out_mat, M, K, N);
        }
        
        free(strides_a);
        free(strides_b);
    }
    
    return out;
}
