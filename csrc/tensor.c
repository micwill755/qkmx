// tensor.c - Core tensor implementation
#include "tensor.h"
#include "util.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

Tensor* tensor_create(int* shape, int ndim, DType dtype) {
    Tensor *t = (Tensor*)malloc(sizeof(Tensor));
    t->ndim = ndim;
    t->dtype = dtype;
    t->shape = (int*)malloc(ndim * sizeof(int));
    t->size = 1;  // Initialize to 1
    
    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        t->size *= shape[i];
    }

    // Calculate bytes once
    size_t bytes;
    switch(dtype) {
        case DTYPE_FLOAT32: bytes = t->size * 4; break;
        case DTYPE_FLOAT16: bytes = t->size * 2; break;
        case DTYPE_INT8:    bytes = t->size * 1; break;
        case DTYPE_INT4:    bytes = (t->size + 1) / 2; break;
        case DTYPE_UINT8:   bytes = t->size * 1; break;
        default:            bytes = t->size * 4; break;
    }
    t->data = calloc(bytes, 1);  // Single allocation

    return t;
}

Tensor* tensor_zeros(int* shape, int ndim, DType dtype) {
    return tensor_create(shape, ndim, dtype); 
}

Tensor* tensor_ones(int* shape, int ndim, DType dtype) {
    Tensor* t = tensor_create(shape, ndim, dtype);
    if (dtype == DTYPE_FLOAT32) {
        float* data = (float*)t->data;
        for (size_t i = 0; i < t->size; i++) {
            data[i] = 1.0f;
        }
    }
    return t;
}

// Create tensor with random normal values
Tensor* tensor_randn(int* shape, int ndim, DType dtype) {
    Tensor* t = tensor_create(shape, ndim, dtype);
    
    // Seed random (call once at program start)
    static int seeded = 0;
    if (!seeded) {
        srand(time(NULL));
        seeded = 1;
    }
    
    // Fill with random values
    if (dtype == DTYPE_FLOAT32) {
        float* data = (float*)t->data;
        for (size_t i = 0; i < t->size; i++) {
            data[i] = randn_float();
        }
    }
    
    return t;
}

// Uniform random [0, 1)
Tensor* tensor_rand(int* shape, int ndim, DType dtype) {
    Tensor* t = tensor_create(shape, ndim, dtype);
    
    static int seeded = 0;
    if (!seeded) {
        srand(time(NULL));
        seeded = 1;
    }
    
    if (dtype == DTYPE_FLOAT32) {
        float* data = (float*)t->data;
        for (size_t i = 0; i < t->size; i++) {
            data[i] = (float)rand() / RAND_MAX;
        }
    }
    
    return t;
}

void tensor_free(Tensor *t) {
    if (t) {
        free(t->data);
        free(t->shape);
        free(t);
    }
}

#define TENSOR_OP_IMPL(type, a, b, out, size, op) \
    do { \
        type* a_data = (type*)(a)->data; \
        type* b_data = (type*)(b)->data; \
        type* out_data = (type*)(out)->data; \
        switch(op) { \
            case TENSOR_MUL: \
                for (size_t i = 0; i < (size); i++) { \
                    out_data[i] = a_data[i] * b_data[i]; \
                } \
                break; \
            case TENSOR_ADD: \
                for (size_t i = 0; i < (size); i++) { \
                    out_data[i] = a_data[i] + b_data[i]; \
                } \
                break; \
        } \
    } while(0)

Tensor* tensor_op(const Tensor* a, const Tensor* b, TensorOperation op) {
    if (a->size != b->size || a->dtype != b->dtype) return NULL;
    
    // TODO: add reference counting or inplace operations 
    Tensor* out = tensor_create(a->shape, a->ndim, a->dtype);

    switch (a->dtype) {
        case DTYPE_FLOAT32: TENSOR_OP_IMPL(float, a, b, out, a->size, op); break;
        case DTYPE_INT8:    TENSOR_OP_IMPL(int8_t, a, b, out, a->size, op); break;
        case DTYPE_UINT8:   TENSOR_OP_IMPL(uint8_t, a, b, out, a->size, op); break;
        default: break;
    }

    return out;
}

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

Tensor* tensor_matmul(const Tensor* a, const Tensor* b) {
    if (a->dtype != b->dtype) return NULL;
    if (a->ndim < 2 || b->ndim < 2) return NULL;
    
    // Get matrix dimensions (last 2 dims)
    int M = a->shape[a->ndim - 2];
    int K = a->shape[a->ndim - 1];
    int K_b = b->shape[b->ndim - 2];
    int N = b->shape[b->ndim - 1];
    
    if (K != K_b) return NULL;  // Inner dims must match
    
    // Determine output ndim (max of the two)
    int out_ndim = (a->ndim > b->ndim) ? a->ndim : b->ndim;
    int batch_ndim = out_ndim - 2;
    
    // Build output shape and check broadcasting compatibility
    int* out_shape = (int*)malloc(out_ndim * sizeof(int));
    
    for (int i = 0; i < batch_ndim; i++) {
        int idx_a = i - (batch_ndim - (a->ndim - 2));
        int idx_b = i - (batch_ndim - (b->ndim - 2));
        
        int dim_a = (idx_a >= 0) ? a->shape[idx_a] : 1;
        int dim_b = (idx_b >= 0) ? b->shape[idx_b] : 1;
        
        // Check broadcasting rule
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
    
    // Calculate total batch size
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
        
        // Calculate strides for batch dimensions
        size_t* strides_a = (size_t*)malloc(batch_ndim * sizeof(size_t));
        size_t* strides_b = (size_t*)malloc(batch_ndim * sizeof(size_t));
        
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
        
        // Perform batched matmul
        for (size_t batch = 0; batch < total_batch; batch++) {
            // Calculate batch indices
            size_t idx_a = 0, idx_b = 0;
            size_t temp = batch;
            
            for (int i = batch_ndim - 1; i >= 0; i--) {
                size_t coord = temp % out->shape[i];
                temp /= out->shape[i];
                idx_a += coord * strides_a[i];
                idx_b += coord * strides_b[i];
            }
            
            // Single matrix multiply
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

Tensor* tensor_transpose(const Tensor* t, int dim0, int dim1) {
    // Validate dimensions
    if (dim0 < 0 || dim0 >= t->ndim || dim1 < 0 || dim1 >= t->ndim) return NULL;
    if (dim0 == dim1) {
        // Same dimension, just copy
        Tensor* out = tensor_create(t->shape, t->ndim, t->dtype);
        memcpy(out->data, t->data, t->size * sizeof(float));  // Assumes FLOAT32
        return out;
    }
    
    // Create new shape with swapped dimensions
    int* new_shape = (int*)malloc(t->ndim * sizeof(int));
    memcpy(new_shape, t->shape, t->ndim * sizeof(int));
    new_shape[dim0] = t->shape[dim1];
    new_shape[dim1] = t->shape[dim0];
    
    Tensor* out = tensor_create(new_shape, t->ndim, t->dtype);
    free(new_shape);
    
    if (t->dtype == DTYPE_FLOAT32) {
        float* in_data = (float*)t->data;
        float* out_data = (float*)out->data;
        
        // Calculate strides
        int* strides = (int*)malloc(t->ndim * sizeof(int));
        strides[t->ndim - 1] = 1;
        for (int i = t->ndim - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * t->shape[i + 1];
        }
        
        // Transpose by iterating through all elements
        int* indices = (int*)malloc(t->ndim * sizeof(int));  // Allocate once
        for (size_t i = 0; i < t->size; i++) {
            size_t idx = i;
            for (int d = 0; d < t->ndim; d++) {
                indices[d] = idx / strides[d];
                idx %= strides[d];
            }
            
            // Swap the dimensions
            int temp = indices[dim0];
            indices[dim0] = indices[dim1];
            indices[dim1] = temp;
            
            // Calculate output index
            size_t out_idx = 0;
            for (int d = 0; d < t->ndim; d++) {
                out_idx = out_idx * out->shape[d] + indices[d];
            }
            
            out_data[out_idx] = in_data[i];
            // NO free here - remove line 211!
        }
        free(indices);  // Free once after loop
        free(strides);
    }
    
    return out;
}

Tensor* tensor_reshape(const Tensor* t, int* new_shape, int new_ndim) {
    // Validate: new shape must have same total size
    size_t new_size = 1;
    for (int i = 0; i < new_ndim; i++) {
        new_size *= new_shape[i];
    }
    if (new_size != t->size) return NULL;

    // Create new tensor (as a view that points to same continguous memory) that shares the data
    Tensor* out = (Tensor*)malloc(sizeof(Tensor));
    out->data = t->data;        // Same pointer - no copy!
    out->dtype = t->dtype;
    out->ndim = new_ndim;
    out->size = t->size;
    out->owns_data = 0;         // Don't free data when this tensor is freed

    // Allocate and copy new shape
    out->shape = (int*)malloc(new_ndim * sizeof(int));
    memcpy(out->shape, new_shape, new_ndim * sizeof(int));

    // Calculate strides for new shape
    out->strides = (int*)malloc(new_ndim * sizeof(int));
    out->strides[new_ndim - 1] = 1;
    for (int i = new_ndim - 2; i >= 0; i--) {
        out->strides[i] = out->strides[i + 1] * out->shape[i + 1];
    }
    
    return out;
}

// tensor.c
Tensor* tensor_layer_norm(const Tensor* x, const Tensor* gamma, 
                          const Tensor* beta, float eps) {
    // Normalize over last dimension
    // x: any shape [..., D]
    // gamma, beta: [D]
    
    Tensor* out = tensor_create(x->shape, x->ndim, x->dtype);
    float* x_data = (float*)x->data;
    float* out_data = (float*)out->data;
    float* gamma_data = (float*)gamma->data;
    float* beta_data = (float*)beta->data;
    
    int D = x->shape[x->ndim - 1];  // Last dimension
    int outer_size = x->size / D;    // All other dimensions
    
    for (int i = 0; i < outer_size; i++) {
        float* row = &x_data[i * D];
        float* out_row = &out_data[i * D];
        
        // Compute mean
        float mean = 0;
        for (int j = 0; j < D; j++) mean += row[j];
        mean /= D;
        
        // Compute variance
        float var = 0;
        for (int j = 0; j < D; j++) {
            float diff = row[j] - mean;
            var += diff * diff;
        }
        var /= D;
        
        // Normalize
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int j = 0; j < D; j++) {
            out_row[j] = (row[j] - mean) * inv_std * gamma_data[j] + beta_data[j];
        }
    }
    
    return out;
}

Tensor* tensor_rms_norm(const Tensor* x, const Tensor* weight, float eps) {
    Tensor* out = tensor_create(x->shape, x->ndim, x->dtype);
    float* x_data = (float*)x->data;
    float* out_data = (float*)out->data;
    float* weight_data = (float*)weight->data;
    
    int D = x->shape[x->ndim - 1];
    int outer_size = x->size / D;
    
    for (int i = 0; i < outer_size; i++) {
        float* row = &x_data[i * D];
        float* out_row = &out_data[i * D];
        
        // Compute mean square
        float ms = 0;
        for (int j = 0; j < D; j++) ms += row[j] * row[j];
        ms /= D;
        
        // Normalize
        float rms = sqrtf(ms + eps);
        for (int j = 0; j < D; j++) {
            out_row[j] = (row[j] / rms) * weight_data[j];
        }
    }
    
    return out;
}

// Create upper triangular mask
Tensor* tensor_triu(int size, int diagonal) {
    int shape[2] = {size, size};
    Tensor* out = tensor_create(shape, 2, DTYPE_FLOAT32);
    float* data = (float*)out->data;
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            data[i * size + j] = (j > i + diagonal - 1) ? 1.0f : 0.0f;
        }
    }
    return out;
}

// Masked fill
Tensor* tensor_masked_fill(const Tensor* t, const Tensor* mask, float value) {
    Tensor* out = tensor_create(t->shape, t->ndim, t->dtype);
    float* in_data = (float*)t->data;
    float* out_data = (float*)out->data;
    float* mask_data = (float*)mask->data;
    
    for (size_t i = 0; i < t->size; i++) {
        out_data[i] = (mask_data[i] != 0.0f) ? value : in_data[i];
    }
    return out;
}

void* tensor_get_data(const Tensor* t) {
    return t->data;
}