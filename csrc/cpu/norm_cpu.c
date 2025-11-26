#include "../tensor.h"
#include <math.h>

Tensor* layer_norm_cpu(const Tensor* x, const Tensor* gamma, const Tensor* beta, float eps) {
    Tensor* out = tensor_create(x->shape, x->ndim, x->dtype, DEVICE_CPU);
    float* x_data = (float*)x->data;
    float* out_data = (float*)out->data;
    float* gamma_data = (float*)gamma->data;
    float* beta_data = (float*)beta->data;
    
    int D = x->shape[x->ndim - 1];
    int outer_size = x->size / D;
    
    for (int i = 0; i < outer_size; i++) {
        float* row = &x_data[i * D];
        float* out_row = &out_data[i * D];
        
        float mean = 0;
        for (int j = 0; j < D; j++) mean += row[j];
        mean /= D;
        
        float var = 0;
        for (int j = 0; j < D; j++) {
            float diff = row[j] - mean;
            var += diff * diff;
        }
        var /= D;
        
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int j = 0; j < D; j++) {
            out_row[j] = (row[j] - mean) * inv_std * gamma_data[j] + beta_data[j];
        }
    }
    
    return out;
}
