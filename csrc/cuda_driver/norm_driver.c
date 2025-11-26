// norm_driver.c - CUDA Driver API layer normalization (pure C)
#include "../tensor.h"
#include "cuda_init.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static CUmodule layernorm_module = NULL;
static CUfunction layernorm_kernel_func = NULL;

// PTX for layer normalization kernel
static const char* layernorm_ptx = 
".version 7.0\n"
".target sm_80\n"
".address_size 64\n"
"\n"
".visible .entry layer_norm_kernel(\n"
"    .param .u64 input,\n"
"    .param .u64 output,\n"
"    .param .u64 gamma,\n"
"    .param .u64 beta,\n"
"    .param .u32 outer_size,\n"
"    .param .u32 D,\n"
"    .param .f32 eps\n"
") {\n"
"    .reg .pred %p<2>;\n"
"    .reg .f32 %f<20>;\n"
"    .reg .b32 %r<15>;\n"
"    .reg .b64 %rd<15>;\n"
"\n"
"    ld.param.u64 %rd1, [input];\n"
"    ld.param.u64 %rd2, [output];\n"
"    ld.param.u64 %rd3, [gamma];\n"
"    ld.param.u64 %rd4, [beta];\n"
"    ld.param.u32 %r1, [outer_size];\n"
"    ld.param.u32 %r2, [D];\n"
"    ld.param.f32 %f1, [eps];\n"
"\n"
"    // idx = blockIdx.x * blockDim.x + threadIdx.x\n"
"    mov.u32 %r3, %ctaid.x;\n"
"    mov.u32 %r4, %ntid.x;\n"
"    mov.u32 %r5, %tid.x;\n"
"    mad.lo.s32 %r6, %r3, %r4, %r5;\n"
"\n"
"    setp.ge.s32 %p1, %r6, %r1;\n"
"    @%p1 bra DONE;\n"
"\n"
"    // Compute mean\n"
"    mov.f32 %f2, 0f00000000;\n"  // mean = 0
"    mov.u32 %r7, 0;\n"
"    mul.lo.s32 %r8, %r6, %r2;\n"  // row_offset = idx * D\n"
"\n"
"MEAN_LOOP:\n"
"    setp.ge.s32 %p1, %r7, %r2;\n"
"    @%p1 bra MEAN_DONE;\n"
"    add.s32 %r9, %r8, %r7;\n"
"    mul.wide.u32 %rd5, %r9, 4;\n"
"    add.s64 %rd6, %rd1, %rd5;\n"
"    ld.global.f32 %f3, [%rd6];\n"
"    add.f32 %f2, %f2, %f3;\n"
"    add.s32 %r7, %r7, 1;\n"
"    bra MEAN_LOOP;\n"
"\n"
"MEAN_DONE:\n"
"    cvt.rn.f32.u32 %f4, %r2;\n"
"    div.rn.f32 %f2, %f2, %f4;\n"  // mean /= D\n"
"\n"
"    // Compute variance\n"
"    mov.f32 %f5, 0f00000000;\n"  // var = 0\n"
"    mov.u32 %r7, 0;\n"
"\n"
"VAR_LOOP:\n"
"    setp.ge.s32 %p1, %r7, %r2;\n"
"    @%p1 bra VAR_DONE;\n"
"    add.s32 %r9, %r8, %r7;\n"
"    mul.wide.u32 %rd5, %r9, 4;\n"
"    add.s64 %rd6, %rd1, %rd5;\n"
"    ld.global.f32 %f3, [%rd6];\n"
"    sub.f32 %f6, %f3, %f2;\n"
"    mul.f32 %f7, %f6, %f6;\n"
"    add.f32 %f5, %f5, %f7;\n"
"    add.s32 %r7, %r7, 1;\n"
"    bra VAR_LOOP;\n"
"\n"
"VAR_DONE:\n"
"    div.rn.f32 %f5, %f5, %f4;\n"  // var /= D\n"
"    add.f32 %f8, %f5, %f1;\n"     // var + eps\n"
"    sqrt.rn.f32 %f9, %f8;\n"      // std = sqrt(var + eps)\n"
"\n"
"    // Normalize\n"
"    mov.u32 %r7, 0;\n"
"\n"
"NORM_LOOP:\n"
"    setp.ge.s32 %p1, %r7, %r2;\n"
"    @%p1 bra DONE;\n"
"    \n"
"    add.s32 %r9, %r8, %r7;\n"
"    mul.wide.u32 %rd5, %r9, 4;\n"
"    add.s64 %rd6, %rd1, %rd5;\n"
"    ld.global.f32 %f3, [%rd6];\n"
"    \n"
"    // normalized = (x - mean) / std\n"
"    sub.f32 %f10, %f3, %f2;\n"
"    div.rn.f32 %f11, %f10, %f9;\n"
"    \n"
"    // Load gamma and beta\n"
"    mul.wide.u32 %rd7, %r7, 4;\n"
"    add.s64 %rd8, %rd3, %rd7;\n"
"    ld.global.f32 %f12, [%rd8];\n"
"    add.s64 %rd9, %rd4, %rd7;\n"
"    ld.global.f32 %f13, [%rd9];\n"
"    \n"
"    // output = gamma * normalized + beta\n"
"    mul.f32 %f14, %f12, %f11;\n"
"    add.f32 %f15, %f14, %f13;\n"
"    \n"
"    add.s64 %rd10, %rd2, %rd5;\n"
"    st.global.f32 [%rd10], %f15;\n"
"    \n"
"    add.s32 %r7, %r7, 1;\n"
"    bra NORM_LOOP;\n"
"\n"
"DONE:\n"
"    ret;\n"
"}\n";

static void init_layernorm_kernel() {
    if (layernorm_module != NULL) return;
    
    cuda_driver_init();  // Shared initialization
    CU_CHECK(cuModuleLoadData(&layernorm_module, layernorm_ptx));
    CU_CHECK(cuModuleGetFunction(&layernorm_kernel_func, layernorm_module, "layer_norm_kernel"));
}

Tensor* layer_norm_cuda_driver(const Tensor* x, const Tensor* gamma, 
                                const Tensor* beta, float eps) {
    init_layernorm_kernel();
    
    int D = x->shape[x->ndim - 1];
    int outer_size = 1;
    for (int i = 0; i < x->ndim - 1; i++) {
        outer_size *= x->shape[i];
    }
    
    size_t x_bytes = x->size * sizeof(float);
    size_t param_bytes = D * sizeof(float);
    
    // Handle inputs based on device
    CUdeviceptr d_input, d_output, d_gamma, d_beta;
    int need_free_input = 0, need_free_gamma = 0, need_free_beta = 0;
    
    if (x->device == DEVICE_CUDA) {
        d_input = (CUdeviceptr)x->data;
    } else {
        CU_CHECK(cuMemAlloc(&d_input, x_bytes));
        CU_CHECK(cuMemcpyHtoD(d_input, x->data, x_bytes));
        need_free_input = 1;
    }
    
    if (gamma->device == DEVICE_CUDA) {
        d_gamma = (CUdeviceptr)gamma->data;
    } else {
        CU_CHECK(cuMemAlloc(&d_gamma, param_bytes));
        CU_CHECK(cuMemcpyHtoD(d_gamma, gamma->data, param_bytes));
        need_free_gamma = 1;
    }
    
    if (beta->device == DEVICE_CUDA) {
        d_beta = (CUdeviceptr)beta->data;
    } else {
        CU_CHECK(cuMemAlloc(&d_beta, param_bytes));
        CU_CHECK(cuMemcpyHtoD(d_beta, beta->data, param_bytes));
        need_free_beta = 1;
    }
    
    CU_CHECK(cuMemAlloc(&d_output, x_bytes));
    
    // Launch kernel
    int threads = 256;
    int blocks = (outer_size + threads - 1) / threads;
    
    void* args[] = { &d_input, &d_output, &d_gamma, &d_beta, &outer_size, &D, &eps };
    
    CU_CHECK(cuLaunchKernel(
        layernorm_kernel_func,
        blocks, 1, 1,
        threads, 1, 1,
        0, NULL, args, NULL
    ));
    
    CU_CHECK(cuCtxSynchronize());
    
    // Create output on same device as input
    Tensor* out = tensor_create(x->shape, x->ndim, x->dtype, x->device);
    if (!out) {
        cuMemFree(d_output);
        if (need_free_input) cuMemFree(d_input);
        if (need_free_gamma) cuMemFree(d_gamma);
        if (need_free_beta) cuMemFree(d_beta);
        return NULL;
    }
    
    // Copy result to output
    if (x->device == DEVICE_CUDA) {
        cuMemcpyDtoD((CUdeviceptr)out->data, d_output, x_bytes);
    } else {
        CU_CHECK(cuMemcpyDtoH(out->data, d_output, x_bytes));
    }
    
    // Free temporary GPU memory
    cuMemFree(d_output);
    if (need_free_input) cuMemFree(d_input);
    if (need_free_gamma) cuMemFree(d_gamma);
    if (need_free_beta) cuMemFree(d_beta);
    
    return out;
}
