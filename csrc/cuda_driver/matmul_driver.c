// matmul_driver.c - CUDA Driver API matrix multiplication (pure C)
#include "../tensor.h"
#include "cuda_init.h"
#include <stdio.h>
#include <stdlib.h>

static CUmodule matmul_module = NULL;
static CUfunction matmul_kernel_func = NULL;

// PTX for simple matmul kernel
static const char* matmul_ptx = 
".version 7.0\n"
".target sm_80\n"
".address_size 64\n"
"\n"
".visible .entry matmul_kernel(\n"
"    .param .u64 A,\n"
"    .param .u64 B,\n"
"    .param .u64 C,\n"
"    .param .u32 M,\n"
"    .param .u32 K,\n"
"    .param .u32 N\n"
") {\n"
"    .reg .pred %p<2>;\n"
"    .reg .f32 %f<10>;\n"
"    .reg .b32 %r<20>;\n"
"    .reg .b64 %rd<20>;\n"
"\n"
"    ld.param.u64 %rd1, [A];\n"
"    ld.param.u64 %rd2, [B];\n"
"    ld.param.u64 %rd3, [C];\n"
"    ld.param.u32 %r1, [M];\n"
"    ld.param.u32 %r2, [K];\n"
"    ld.param.u32 %r3, [N];\n"
"\n"
"    // row = blockIdx.y * blockDim.y + threadIdx.y\n"
"    mov.u32 %r4, %ctaid.y;\n"
"    mov.u32 %r5, %ntid.y;\n"
"    mov.u32 %r6, %tid.y;\n"
"    mad.lo.s32 %r7, %r4, %r5, %r6;\n"
"\n"
"    // col = blockIdx.x * blockDim.x + threadIdx.x\n"
"    mov.u32 %r8, %ctaid.x;\n"
"    mov.u32 %r9, %ntid.x;\n"
"    mov.u32 %r10, %tid.x;\n"
"    mad.lo.s32 %r11, %r8, %r9, %r10;\n"
"\n"
"    // Check bounds\n"
"    setp.ge.s32 %p1, %r7, %r1;\n"
"    @%p1 bra DONE;\n"
"    setp.ge.s32 %p1, %r11, %r3;\n"
"    @%p1 bra DONE;\n"
"\n"
"    // sum = 0\n"
"    mov.f32 %f1, 0f00000000;\n"
"    mov.u32 %r12, 0;\n"
"\n"
"LOOP:\n"
"    setp.ge.s32 %p1, %r12, %r2;\n"
"    @%p1 bra LOOP_END;\n"
"\n"
"    // A[row * K + k]\n"
"    mad.lo.s32 %r13, %r7, %r2, %r12;\n"
"    mul.wide.u32 %rd4, %r13, 4;\n"
"    add.s64 %rd5, %rd1, %rd4;\n"
"    ld.global.f32 %f2, [%rd5];\n"
"\n"
"    // B[k * N + col]\n"
"    mad.lo.s32 %r14, %r12, %r3, %r11;\n"
"    mul.wide.u32 %rd6, %r14, 4;\n"
"    add.s64 %rd7, %rd2, %rd6;\n"
"    ld.global.f32 %f3, [%rd7];\n"
"\n"
"    // sum += A * B\n"
"    fma.rn.f32 %f1, %f2, %f3, %f1;\n"
"\n"
"    add.s32 %r12, %r12, 1;\n"
"    bra LOOP;\n"
"\n"
"LOOP_END:\n"
"    // C[row * N + col] = sum\n"
"    mad.lo.s32 %r15, %r7, %r3, %r11;\n"
"    mul.wide.u32 %rd8, %r15, 4;\n"
"    add.s64 %rd9, %rd3, %rd8;\n"
"    st.global.f32 [%rd9], %f1;\n"
"\n"
"DONE:\n"
"    ret;\n"
"}\n";

static void init_matmul_kernel() {
    if (matmul_module != NULL) return;
    
    cuda_driver_init();  // Shared initialization
    
    CU_CHECK(cuModuleLoadData(&matmul_module, matmul_ptx));
    CU_CHECK(cuModuleGetFunction(&matmul_kernel_func, matmul_module, "matmul_kernel"));
}

Tensor* matmul_cuda_driver(const Tensor* a, const Tensor* b) {
    printf("matmul_cuda\n");  // Debug: show we're using GPU
    init_matmul_kernel();
    if (a->dtype != b->dtype) return NULL;
    if (a->ndim < 2 || b->ndim < 2) return NULL;
    
    int M = a->shape[a->ndim - 2];
    int K = a->shape[a->ndim - 1];
    int K_b = b->shape[b->ndim - 2];
    int N = b->shape[b->ndim - 1];
    
    if (K != K_b) return NULL;
    
    // Only 2D for now
    if (a->ndim != 2 || b->ndim != 2) {
        fprintf(stderr, "CUDA Driver matmul: only 2D matrices supported\n");
        return NULL;
    }
    
    size_t a_bytes = M * K * sizeof(float);
    size_t b_bytes = K * N * sizeof(float);
    size_t c_bytes = M * N * sizeof(float);
    
    // Handle inputs based on device
    CUdeviceptr d_A, d_B, d_C;
    int need_free_a = 0, need_free_b = 0;
    
    if (a->device == DEVICE_CUDA) {
        d_A = (CUdeviceptr)a->data;
    } else {
        CU_CHECK(cuMemAlloc(&d_A, a_bytes));
        CU_CHECK(cuMemcpyHtoD(d_A, a->data, a_bytes));
        need_free_a = 1;
    }
    
    if (b->device == DEVICE_CUDA) {
        d_B = (CUdeviceptr)b->data;
    } else {
        CU_CHECK(cuMemAlloc(&d_B, b_bytes));
        CU_CHECK(cuMemcpyHtoD(d_B, b->data, b_bytes));
        need_free_b = 1;
    }
    
    CU_CHECK(cuMemAlloc(&d_C, c_bytes));
    
    // Launch kernel with 2D grid
    int threads_x = 16, threads_y = 16;
    int blocks_x = (N + threads_x - 1) / threads_x;
    int blocks_y = (M + threads_y - 1) / threads_y;
    
    void* args[] = { &d_A, &d_B, &d_C, &M, &K, &N };
    
    CU_CHECK(cuLaunchKernel(
        matmul_kernel_func,
        blocks_x, blocks_y, 1,
        threads_x, threads_y, 1,
        0, NULL, args, NULL
    ));
    
    CU_CHECK(cuCtxSynchronize());
    
    // Create output on same device as inputs
    int out_shape[2] = {M, N};
    Tensor* out = tensor_create(out_shape, 2, a->dtype, a->device);
    if (!out) {
        cuMemFree(d_C);
        if (need_free_a) cuMemFree(d_A);
        if (need_free_b) cuMemFree(d_B);
        return NULL;
    }
    
    // Copy result to output
    if (a->device == DEVICE_CUDA) {
        cuMemcpyDtoD((CUdeviceptr)out->data, d_C, c_bytes);
    } else {
        CU_CHECK(cuMemcpyDtoH(out->data, d_C, c_bytes));
    }
    
    // Free temporary GPU memory
    cuMemFree(d_C);
    if (need_free_a) cuMemFree(d_A);
    if (need_free_b) cuMemFree(d_B);
    
    return out;
}
