// tensor.c - Core tensor implementation (dispatch layer)
#include "tensor.h"
#include "util.h"
#include "cpu/tensor_cpu.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdio.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#ifdef USE_CUDA
#include "cuda_driver/tensor_cuda.h"
#include <cuda.h>
#endif

#include "dispatch.h"

// ============================================================================
// MEMORY MANAGEMENT - Dispatch to backend
// ============================================================================

Tensor* tensor_create(int* shape, int ndim, DType dtype, DeviceType device) {
    if (device == DEVICE_CUDA) {
#ifdef USE_CUDA
        return tensor_create_cuda(shape, ndim, dtype);
#else
        fprintf(stderr, "CUDA not available, falling back to CPU\n");
        return tensor_create_cpu(shape, ndim, dtype);
#endif
    }
    return tensor_create_cpu(shape, ndim, dtype);
}

Tensor* tensor_zeros(int* shape, int ndim, DType dtype, DeviceType device) {
    return tensor_create(shape, ndim, dtype, device);
}

// ============================================================================
// TENSOR CREATION
// ============================================================================

Tensor* tensor_ones(int* shape, int ndim, DType dtype, DeviceType device) {
    Tensor* t = tensor_create(shape, ndim, dtype, DEVICE_CPU);  // Create on CPU first
    if (dtype == DTYPE_FLOAT32) {
        float* data = (float*)t->data;
        for (size_t i = 0; i < t->size; i++) {
            data[i] = 1.0f;
        }
    }
    // Move to requested device if needed
    if (device == DEVICE_CUDA) {
        Tensor* gpu_t = tensor_to_cuda(t);
        tensor_free(t);
        return gpu_t;
    }
    return t;
}

static void ensure_seeded(void) {
    static int seeded = 0;
    if (!seeded) {
        srand(time(NULL));
        seeded = 1;
    }
}

Tensor* tensor_randn(int* shape, int ndim, DType dtype, DeviceType device) {
    Tensor* t = tensor_create(shape, ndim, dtype, DEVICE_CPU);  // Generate on CPU
    if (!t) return NULL;
    
    ensure_seeded();
    
    if (dtype == DTYPE_FLOAT32) {
        float* data = (float*)t->data;
        for (size_t i = 0; i < t->size; i++) {
            data[i] = randn_float();
        }
    }
    
    if (device == DEVICE_CUDA) {
        Tensor* gpu_t = tensor_to_cuda(t);
        tensor_free(t);
        return gpu_t;
    }
    return t;
}

Tensor* tensor_rand(int* shape, int ndim, DType dtype, DeviceType device) {
    Tensor* t = tensor_create(shape, ndim, dtype, DEVICE_CPU);  // Generate on CPU
    if (!t) return NULL;
    
    ensure_seeded();
    
    if (dtype == DTYPE_FLOAT32) {
        float* data = (float*)t->data;
        for (size_t i = 0; i < t->size; i++) {
            data[i] = (float)rand() / RAND_MAX;
        }
    }
    
    if (device == DEVICE_CUDA) {
        Tensor* gpu_t = tensor_to_cuda(t);
        tensor_free(t);
        return gpu_t;
    }
    return t;
}

void tensor_free(Tensor *t) {
    if (!t) return;
    
    if (t->device == DEVICE_CUDA) {
#ifdef USE_CUDA
        tensor_free_cuda(t);
#endif
    } else {
        tensor_free_cpu(t);
    }
}


// ============================================================================
// ELEMENT-WISE OPERATIONS
// ============================================================================

Tensor* tensor_op(const Tensor* a, const Tensor* b, TensorOperation op) {
    if (a->dtype != b->dtype) return NULL;
    
    int out_ndim = (a->ndim > b->ndim) ? a->ndim : b->ndim;
    int* out_shape = malloc(out_ndim * sizeof(int));
    if (!out_shape) return NULL;
    
    for (int i = 0; i < out_ndim; i++) {
        int idx_a = i - (out_ndim - a->ndim);
        int idx_b = i - (out_ndim - b->ndim);
        
        int dim_a = (idx_a >= 0) ? a->shape[idx_a] : 1;
        int dim_b = (idx_b >= 0) ? b->shape[idx_b] : 1;
        
        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            free(out_shape);
            return NULL;
        }
        out_shape[i] = (dim_a > dim_b) ? dim_a : dim_b;
    }
    
    Tensor* out = tensor_create(out_shape, out_ndim, a->dtype, a->device);
    free(out_shape);
    
    if (a->dtype == DTYPE_FLOAT32) {
        float* a_data = (float*)a->data;
        float* b_data = (float*)b->data;
        float* out_data = (float*)out->data;
        
        size_t* a_strides = malloc(out_ndim * sizeof(size_t));
        size_t* b_strides = malloc(out_ndim * sizeof(size_t));
        if (!a_strides || !b_strides) {
            free(a_strides);
            free(b_strides);
            tensor_free(out);
            return NULL;
        }
        
        a_strides[out_ndim - 1] = 1;
        b_strides[out_ndim - 1] = 1;
        
        for (int d = out_ndim - 2; d >= 0; d--) {
            int idx_a = d + 1 - (out_ndim - a->ndim);
            int idx_b = d + 1 - (out_ndim - b->ndim);
            
            int dim_a_next = (idx_a >= 0) ? a->shape[idx_a] : 1;
            int dim_b_next = (idx_b >= 0) ? b->shape[idx_b] : 1;
            
            a_strides[d] = a_strides[d + 1] * dim_a_next;
            b_strides[d] = b_strides[d + 1] * dim_b_next;
        }
        
        // Broadcast operation
        for (size_t i = 0; i < out->size; i++) {
            size_t idx = i;
            size_t a_idx = 0, b_idx = 0;
            
            for (int d = out_ndim - 1; d >= 0; d--) {
                int coord = idx % out->shape[d];
                idx /= out->shape[d];
                
                int idx_a = d - (out_ndim - a->ndim);
                int idx_b = d - (out_ndim - b->ndim);
                
                if (idx_a >= 0 && a->shape[idx_a] > 1) {
                    a_idx += coord * a_strides[d];
                }
                if (idx_b >= 0 && b->shape[idx_b] > 1) {
                    b_idx += coord * b_strides[d];
                }
            }
            
            switch(op) {
                case TENSOR_ADD:
                    out_data[i] = a_data[a_idx] + b_data[b_idx];
                    break;
                case TENSOR_MUL:
                    out_data[i] = a_data[a_idx] * b_data[b_idx];
                    break;
                case TENSOR_SUB:
                    out_data[i] = a_data[a_idx] - b_data[b_idx];
                    break;
            }
        }
        
        free(a_strides);
        free(b_strides);
    }
    
    return out;
}

Tensor* tensor_scalar_op(const Tensor* t, float scalar, TensorOperation op) {
    Tensor* out = tensor_create(t->shape, t->ndim, t->dtype, t->device);
    
    if (t->dtype == DTYPE_FLOAT32) {
        float* in_data = (float*)t->data;
        float* out_data = (float*)out->data;
        
        for (size_t i = 0; i < t->size; i++) {
            switch(op) {
                case TENSOR_MUL:
                    out_data[i] = in_data[i] * scalar;
                    break;
                case TENSOR_ADD:
                    out_data[i] = in_data[i] + scalar;
                    break;
                case TENSOR_DIV:
                    out_data[i] = in_data[i] / scalar;
                    break;
            }
        }
    }
    
    return out;
}

// ============================================================================
// MATRIX OPERATIONS
// ============================================================================

Tensor* tensor_matmul(const Tensor* a, const Tensor* b) {
    // Ensure both tensors on same device
    if (a->device != b->device) {
        fprintf(stderr, "Tensors must be on same device\n");
        return NULL;
    }
    const DispatchTable* dispatch = get_dispatch_table(a->device);
    return dispatch->matmul(a, b);
}

Tensor* tensor_transpose(const Tensor* t, int dim0, int dim1) {
    // Validate dimensions
    if (dim0 < 0 || dim0 >= t->ndim || dim1 < 0 || dim1 >= t->ndim) return NULL;
    if (dim0 == dim1) {
        // Same dimension, just copy
        Tensor* out = tensor_create(t->shape, t->ndim, t->dtype, t->device);
        memcpy(out->data, t->data, t->size * sizeof(float));  // Assumes FLOAT32
        return out;
    }
    
    // Create new shape with swapped dimensions
    int* new_shape = (int*)malloc(t->ndim * sizeof(int));
    memcpy(new_shape, t->shape, t->ndim * sizeof(int));
    new_shape[dim0] = t->shape[dim1];
    new_shape[dim1] = t->shape[dim0];
    
    Tensor* out = tensor_create(new_shape, t->ndim, t->dtype, t->device);
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

// ============================================================================
// NORMALIZATION
// ============================================================================

Tensor* tensor_layer_norm(const Tensor* x, const Tensor* gamma, 
                          const Tensor* beta, float eps) {
    const DispatchTable* dispatch = get_dispatch_table(x->device);
    return dispatch->layer_norm(x, gamma, beta, eps);
}

Tensor* tensor_rms_norm(const Tensor* x, const Tensor* weight, float eps) {
    Tensor* out = tensor_create(x->shape, x->ndim, x->dtype, x->device);
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

// ============================================================================
// MASKING & UTILITIES
// ============================================================================

Tensor* tensor_triu(int size, int diagonal) {
    int shape[2] = {size, size};
    Tensor* out = tensor_create(shape, 2, DTYPE_FLOAT32, DEVICE_CPU);
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
    Tensor* out = tensor_create(t->shape, t->ndim, t->dtype, DEVICE_CPU);
    float* in_data = (float*)t->data;
    float* out_data = (float*)out->data;
    float* mask_data = (float*)mask->data;
    
    for (size_t i = 0; i < t->size; i++) {
        out_data[i] = (mask_data[i] != 0.0f) ? value : in_data[i];
    }
    return out;
}

Tensor* tensor_softmax(const Tensor* t, int dim) {
    const DispatchTable* dispatch = get_dispatch_table(t->device);
    return dispatch->softmax(t, dim);
}

// ============================================================================
// ACTIVATION FUNCTIONS
// ============================================================================

Tensor* tensor_gelu(const Tensor* t) {
    const DispatchTable* dispatch = get_dispatch_table(t->device);
    return dispatch->gelu(t);
}

// ============================================================================
// INDEXING & SLICING
// ============================================================================

Tensor* tensor_get_index(Tensor* t, int index) {
    if (t->ndim == 1) {
        // Return scalar as 0-d tensor (must copy for scalar)
        int shape = 1;
        Tensor* result = tensor_create(&shape, 1, t->dtype, DEVICE_CPU);

        float* src = (float*)t->data + index;
        float* dst = (float*)result->data;
        *dst = *src;
        return result;
    }
    
    // Calculate slice size
    int slice_size = 1;
    for (int i = 1; i < t->ndim; i++) {
        slice_size *= t->shape[i];
    }
    
    // Create view 
    Tensor* result = (Tensor*)malloc(sizeof(Tensor));
    result->ndim = t->ndim - 1;
    result->dtype = t->dtype;
    result->size = slice_size;
    result->owns_data = 0;  // Don't free data - it's shared!
    
    // Point to slice location in original tensor
    result->data = (float*)t->data + (index * slice_size);
    
    // Copy shape (excluding first dimension)
    result->shape = (int*)malloc(result->ndim * sizeof(int));
    memcpy(result->shape, t->shape + 1, result->ndim * sizeof(int));
    
    // Calculate strides
    result->strides = (int*)malloc(result->ndim * sizeof(int));
    result->strides[result->ndim - 1] = 1;
    for (int i = result->ndim - 2; i >= 0; i--) {
        result->strides[i] = result->strides[i + 1] * result->shape[i + 1];
    }
    
    return result;
}

// Set tensor slice at index
int tensor_set_index(Tensor* t, int index, Tensor* value) {
    int slice_size = 1;
    for (int i = 1; i < t->ndim; i++) {
        slice_size *= t->shape[i];
    }
    
    float* dst = (float*)t->data + (index * slice_size);
    memcpy(dst, value->data, slice_size * sizeof(float));
    
    return 0;
}

// Advanced indexing: gather rows based on tensor of indices
Tensor* tensor_advanced_index(Tensor* t, Tensor* indices) {
    // indices can be any shape, output will be indices.shape + t.shape[1:]
    int* out_shape = (int*)malloc((indices->ndim + t->ndim - 1) * sizeof(int));
    
    // Copy indices shape
    for (int i = 0; i < indices->ndim; i++) {
        out_shape[i] = indices->shape[i];
    }
    
    // Copy remaining dimensions from t
    for (int i = 1; i < t->ndim; i++) {
        out_shape[indices->ndim + i - 1] = t->shape[i];
    }
    
    int out_ndim = indices->ndim + t->ndim - 1;
    Tensor* result = tensor_create(out_shape, out_ndim, t->dtype, DEVICE_CPU);
    free(out_shape);
    
    // Calculate row size
    int row_size = 1;
    for (int i = 1; i < t->ndim; i++) {
        row_size *= t->shape[i];
    }
    
    // Gather rows
    float* t_data = (float*)t->data;
    float* indices_data = (float*)indices->data;
    float* result_data = (float*)result->data;
    
    for (size_t i = 0; i < indices->size; i++) {
        int idx = (int)indices_data[i];
        
        // Bounds check
        if (idx < 0 || idx >= t->shape[0]) {
            tensor_free(result);
            return NULL;
        }
        
        // Copy row
        memcpy(result_data + i * row_size, 
               t_data + idx * row_size, 
               row_size * sizeof(float));
    }
    
    return result;
}

int tensor_set_scalar(Tensor* t, int index, float value) {
    if (t->ndim == 1) {
        ((float*)t->data)[index] = value;
        return 0;
    }
    
    int slice_size = 1;
    for (int i = 1; i < t->ndim; i++) {
        slice_size *= t->shape[i];
    }
    
    float* dst = (float*)t->data + (index * slice_size);
    for (int i = 0; i < slice_size; i++) {
        dst[i] = value;
    }
    
    return 0;
}

// Get scalar value from tensor
float tensor_get_scalar(Tensor* t, int* indices, int num_indices) {
    if (t->dtype != DTYPE_FLOAT32) return 0.0f;
    
    // Calculate flat index
    size_t flat_idx = 0;
    size_t stride = 1;
    for (int i = t->ndim - 1; i >= 0; i--) {
        int idx = (i < num_indices) ? indices[i] : 0;
        flat_idx += idx * stride;
        stride *= t->shape[i];
    }
    
    return ((float*)t->data)[flat_idx];
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

void* tensor_get_data(const Tensor* t) {
    return t->data;
}

float tensor_mean(const Tensor* t) {
    if (t->dtype != DTYPE_FLOAT32) return 0.0f;
    
    float* data = (float*)t->data;
    float sum = 0.0f;
    for (size_t i = 0; i < t->size; i++) {
        sum += data[i];
    }
    return sum / t->size;
}

// ============================================================================
// DEVICE TRANSFER
// ============================================================================

Tensor* tensor_to_cuda(const Tensor* t) {
#ifdef USE_CUDA
    if (t->device == DEVICE_CUDA) {
        // Already on CUDA - make a copy
        Tensor* out = tensor_create_cuda(t->shape, t->ndim, t->dtype);
        if (!out) return NULL;
        size_t bytes = t->size * sizeof(float);
        cuMemcpyDtoD((CUdeviceptr)out->data, (CUdeviceptr)t->data, bytes);
        return out;
    }
    // CPU to CUDA
    return tensor_cpu_to_cuda(t);
#else
    fprintf(stderr, "CUDA not available\n");
    return NULL;
#endif
}

Tensor* tensor_to_cpu(const Tensor* t) {
    if (t->device == DEVICE_CPU) {
        // Already on CPU - make a copy
        Tensor* out = tensor_create_cpu(t->shape, t->ndim, t->dtype);
        if (!out) return NULL;
        memcpy(out->data, t->data, t->size * sizeof(float));
        return out;
    }
#ifdef USE_CUDA
    // CUDA to CPU
    return tensor_cuda_to_cpu(t);
#else
    fprintf(stderr, "CUDA not available\n");
    return NULL;
#endif
}