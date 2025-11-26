// activation_driver.c - CUDA Driver API implementation (pure C)
#include "../tensor.h"
#include "cuda_init.h"
#include <stdio.h>
#include <stdlib.h>

// Global module and function handles (initialized once)
static CUmodule gelu_module = NULL;
static CUfunction gelu_kernel_func = NULL;

// PTX code for GELU kernel (embedded as string)
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
static const char* gelu_ptx = 
".version 7.0\n"
".target sm_80\n"
".address_size 64\n"
"\n"
".visible .entry gelu_kernel(\n"
"    .param .u64 input,\n"
"    .param .u64 output,\n"
"    .param .u64 size\n"
") {\n"
"    .reg .pred %p<2>;\n"
"    .reg .f32 %f<16>;\n"
"    .reg .b32 %r<5>;\n"
"    .reg .b64 %rd<10>;\n"
"\n"
"    ld.param.u64 %rd1, [input];\n"
"    ld.param.u64 %rd2, [output];\n"
"    ld.param.u64 %rd3, [size];\n"
"\n"
"    mov.u32 %r1, %ctaid.x;\n"
"    mov.u32 %r2, %ntid.x;\n"
"    mov.u32 %r3, %tid.x;\n"
"    mad.lo.s32 %r4, %r1, %r2, %r3;\n"
"\n"
"    cvt.u64.u32 %rd4, %r4;\n"
"    setp.ge.u64 %p1, %rd4, %rd3;\n"
"    @%p1 bra DONE;\n"
"\n"
"    mul.wide.u32 %rd5, %r4, 4;\n"
"    add.s64 %rd6, %rd1, %rd5;\n"
"    ld.global.f32 %f1, [%rd6];\n"
"\n"
"    // x^2\n"
"    mul.f32 %f2, %f1, %f1;\n"
"    // x^3\n"
"    mul.f32 %f3, %f2, %f1;\n"
"    // 0.044715 * x^3\n"
"    mov.f32 %f4, 0f3D372713;\n"
"    mul.f32 %f5, %f4, %f3;\n"
"    // x + 0.044715 * x^3\n"
"    add.f32 %f6, %f1, %f5;\n"
"    // sqrt(2/pi) * (x + 0.044715 * x^3)\n"
"    mov.f32 %f7, 0f3F4C422A;\n"
"    mul.f32 %f8, %f7, %f6;\n"
"    // tanh approximation using ex2 (e^x)\n"
"    // tanh(x) = (e^2x - 1) / (e^2x + 1)\n"
"    mul.f32 %f9, %f8, 0f40000000;\n"  // 2*x for e^(2x)\n"
"    mul.f32 %f9, %f9, 0f3FB8AA3B;\n"  // multiply by log2(e) = 1.4427\n"
"    ex2.approx.f32 %f10, %f9;\n"      // e^(2x)\n"
"    add.f32 %f11, %f10, 0f3F800000;\n" // e^(2x) + 1\n"
"    sub.f32 %f12, %f10, 0f3F800000;\n" // e^(2x) - 1\n"
"    div.approx.f32 %f13, %f12, %f11;\n" // tanh = (e^2x-1)/(e^2x+1)\n"
"    // 1 + tanh(...)\n"
"    add.f32 %f14, %f13, 0f3F800000;\n"
"    // x * (1 + tanh(...))\n"
"    mul.f32 %f15, %f1, %f14;\n"
"    // 0.5 * x * (1 + tanh(...))\n"
"    mul.f32 %f15, %f15, 0f3F000000;\n"
"\n"
"    add.s64 %rd7, %rd2, %rd5;\n"
"    st.global.f32 [%rd7], %f15;\n"
"\n"
"DONE:\n"
"    ret;\n"
"}\n";

// Initialize CUDA Driver API and load kernel
static void init_gelu_kernel() {
    if (gelu_module != NULL) return;  // Already initialized
    
    cuda_driver_init();  // Shared initialization
    CU_CHECK(cuModuleLoadData(&gelu_module, gelu_ptx));
    CU_CHECK(cuModuleGetFunction(&gelu_kernel_func, gelu_module, "gelu_kernel"));
}

Tensor* gelu_cuda_driver(const Tensor* t) {
    init_gelu_kernel();
    
    size_t bytes = t->size * sizeof(float);
    
    // Allocate GPU memory for output
    CUdeviceptr d_input, d_output;
    CU_CHECK(cuMemAlloc(&d_output, bytes));
    
    int need_free_input = 0;
    
    // Handle input based on device
    if (t->device == DEVICE_CUDA) {
        // Input already on GPU - use directly
        d_input = (CUdeviceptr)t->data;
    } else {
        // Input on CPU - copy to GPU
        CU_CHECK(cuMemAlloc(&d_input, bytes));
        CU_CHECK(cuMemcpyHtoD(d_input, t->data, bytes));
        need_free_input = 1;
    }
    
    // Launch kernel
    int threads = 256;
    int blocks = (t->size + threads - 1) / threads;
    size_t size = t->size;
    
    void* args[] = { &d_input, &d_output, &size };
    
    CU_CHECK(cuLaunchKernel(
        gelu_kernel_func,
        blocks, 1, 1,      // grid dimensions
        threads, 1, 1,     // block dimensions
        0,                 // shared memory
        NULL,              // stream
        args,              // kernel arguments
        NULL               // extra
    ));
    
    // Wait for completion
    CU_CHECK(cuCtxSynchronize());
    
    // Create output tensor on same device as input
    Tensor* out = tensor_create(t->shape, t->ndim, t->dtype, t->device);
    if (!out) {
        cuMemFree(d_output);
        if (need_free_input) cuMemFree(d_input);
        return NULL;
    }
    
    // Copy result to output tensor
    if (t->device == DEVICE_CUDA) {
        // Output stays on GPU
        cuMemcpyDtoD((CUdeviceptr)out->data, d_output, bytes);
    } else {
        // Copy back to CPU
        CU_CHECK(cuMemcpyDtoH(out->data, d_output, bytes));
    }
    
    // Free temporary GPU memory
    cuMemFree(d_output);
    if (need_free_input) cuMemFree(d_input);
    
    return out;
}

// Softmax PTX kernel - one thread per row, handles last dimension
static CUmodule softmax_module = NULL;
static CUfunction softmax_kernel_func = NULL;

static const char* softmax_ptx = 
".version 7.0\n"
".target sm_80\n"
".address_size 64\n"
"\n"
".visible .entry softmax_kernel(\n"
"    .param .u64 input,\n"
"    .param .u64 output,\n"
"    .param .u32 outer_size,\n"
"    .param .u32 dim_size\n"
") {\n"
"    .reg .pred %p<2>;\n"
"    .reg .f32 %f<8>;\n"
"    .reg .b32 %r<10>;\n"
"    .reg .b64 %rd<10>;\n"
"\n"
"    ld.param.u64 %rd1, [input];\n"
"    ld.param.u64 %rd2, [output];\n"
"    ld.param.u32 %r1, [outer_size];\n"
"    ld.param.u32 %r2, [dim_size];\n"
"\n"
"    // row = blockIdx.x * blockDim.x + threadIdx.x\n"
"    mov.u32 %r3, %ctaid.x;\n"
"    mov.u32 %r4, %ntid.x;\n"
"    mov.u32 %r5, %tid.x;\n"
"    mad.lo.s32 %r6, %r3, %r4, %r5;\n"
"\n"
"    setp.ge.s32 %p1, %r6, %r1;\n"
"    @%p1 bra DONE;\n"
"\n"
"    // base_offset = row * dim_size * 4\n"
"    mul.lo.s32 %r7, %r6, %r2;\n"
"    mul.wide.u32 %rd3, %r7, 4;\n"
"    add.s64 %rd4, %rd1, %rd3;\n"  // input row ptr
"    add.s64 %rd5, %rd2, %rd3;\n"  // output row ptr
"\n"
"    // Find max in row\n"
"    ld.global.f32 %f1, [%rd4];\n"  // max = input[0]
"    mov.u32 %r8, 1;\n"
"MAX_LOOP:\n"
"    setp.ge.s32 %p1, %r8, %r2;\n"
"    @%p1 bra MAX_DONE;\n"
"    mul.wide.u32 %rd6, %r8, 4;\n"
"    add.s64 %rd7, %rd4, %rd6;\n"
"    ld.global.f32 %f2, [%rd7];\n"
"    max.f32 %f1, %f1, %f2;\n"
"    add.s32 %r8, %r8, 1;\n"
"    bra MAX_LOOP;\n"
"MAX_DONE:\n"
"\n"
"    // Compute exp(x - max) and sum\n"
"    mov.f32 %f3, 0f00000000;\n"  // sum = 0
"    mov.u32 %r8, 0;\n"
"EXP_LOOP:\n"
"    setp.ge.s32 %p1, %r8, %r2;\n"
"    @%p1 bra EXP_DONE;\n"
"    mul.wide.u32 %rd6, %r8, 4;\n"
"    add.s64 %rd7, %rd4, %rd6;\n"
"    ld.global.f32 %f2, [%rd7];\n"
"    sub.f32 %f4, %f2, %f1;\n"      // x - max
"    mul.f32 %f4, %f4, 0f3FB8AA3B;\n"  // * log2(e)\n"
"    ex2.approx.f32 %f5, %f4;\n"    // exp(x - max)
"    add.s64 %rd8, %rd5, %rd6;\n"
"    st.global.f32 [%rd8], %f5;\n"  // store exp temporarily
"    add.f32 %f3, %f3, %f5;\n"      // sum += exp
"    add.s32 %r8, %r8, 1;\n"
"    bra EXP_LOOP;\n"
"EXP_DONE:\n"
"\n"
"    // Divide by sum\n"
"    mov.u32 %r8, 0;\n"
"DIV_LOOP:\n"
"    setp.ge.s32 %p1, %r8, %r2;\n"
"    @%p1 bra DONE;\n"
"    mul.wide.u32 %rd6, %r8, 4;\n"
"    add.s64 %rd8, %rd5, %rd6;\n"
"    ld.global.f32 %f5, [%rd8];\n"
"    div.approx.f32 %f6, %f5, %f3;\n"
"    st.global.f32 [%rd8], %f6;\n"
"    add.s32 %r8, %r8, 1;\n"
"    bra DIV_LOOP;\n"
"\n"
"DONE:\n"
"    ret;\n"
"}\n";

static void init_softmax_kernel() {
    if (softmax_module != NULL) return;
    
    cuda_driver_init();
    CU_CHECK(cuModuleLoadData(&softmax_module, softmax_ptx));
    CU_CHECK(cuModuleGetFunction(&softmax_kernel_func, softmax_module, "softmax_kernel"));
}

Tensor* softmax_cuda_driver(const Tensor* t, int dim) {
    init_softmax_kernel();
    
    // Handle negative dim
    if (dim < 0) dim += t->ndim;
    
    // For now, only support last dimension
    if (dim != t->ndim - 1) {
        fprintf(stderr, "softmax_cuda_driver: only last dimension supported\n");
        return NULL;
    }
    
    int dim_size = t->shape[dim];
    int outer_size = t->size / dim_size;
    
    size_t bytes = t->size * sizeof(float);
    
    // Handle input based on device
    CUdeviceptr d_input, d_output;
    int need_free_input = 0;
    
    if (t->device == DEVICE_CUDA) {
        d_input = (CUdeviceptr)t->data;
    } else {
        CU_CHECK(cuMemAlloc(&d_input, bytes));
        CU_CHECK(cuMemcpyHtoD(d_input, t->data, bytes));
        need_free_input = 1;
    }
    
    CU_CHECK(cuMemAlloc(&d_output, bytes));
    
    // Launch kernel - one thread per row
    int threads = 256;
    int blocks = (outer_size + threads - 1) / threads;
    
    void* args[] = { &d_input, &d_output, &outer_size, &dim_size };
    
    CU_CHECK(cuLaunchKernel(
        softmax_kernel_func,
        blocks, 1, 1,
        threads, 1, 1,
        0, NULL, args, NULL
    ));
    
    CU_CHECK(cuCtxSynchronize());
    
    // Create output tensor on same device as input
    Tensor* out = tensor_create(t->shape, t->ndim, t->dtype, t->device);
    if (!out) {
        cuMemFree(d_output);
        if (need_free_input) cuMemFree(d_input);
        return NULL;
    }
    
    // Copy result to output
    if (t->device == DEVICE_CUDA) {
        cuMemcpyDtoD((CUdeviceptr)out->data, d_output, bytes);
    } else {
        CU_CHECK(cuMemcpyDtoH(out->data, d_output, bytes));
    }
    
    // Free temporary GPU memory
    cuMemFree(d_output);
    if (need_free_input) cuMemFree(d_input);
    
    return out;
}
