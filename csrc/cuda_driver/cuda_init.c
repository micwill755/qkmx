// cuda_init.c - Shared CUDA Driver API initialization
#include "cuda_init.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static CUcontext g_cuda_context = NULL;
static int g_cuda_initialized = 0;

void cuda_driver_init(void) {
    if (g_cuda_initialized) return;
    
    CUresult err = cuInit(0);
    if (err != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(err, &errStr);
        fprintf(stderr, "Failed to initialize CUDA: %s\n", errStr);
        exit(1);
    }
    
    // Get device 0
    CUdevice device;
    err = cuDeviceGet(&device, 0);
    if (err != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(err, &errStr);
        fprintf(stderr, "Failed to get CUDA device: %s\n", errStr);
        exit(1);
    }
    
    // Use primary context - more compatible across CUDA versions
    err = cuDevicePrimaryCtxRetain(&g_cuda_context, device);
    if (err != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(err, &errStr);
        fprintf(stderr, "Failed to retain primary context: %s\n", errStr);
        exit(1);
    }
    
    // Set as current context
    err = cuCtxSetCurrent(g_cuda_context);
    if (err != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(err, &errStr);
        fprintf(stderr, "Failed to set current context: %s\n", errStr);
        exit(1);
    }
    
    g_cuda_initialized = 1;
    
    // Print GPU info
    char name[256];
    cuDeviceGetName(name, sizeof(name), device);
    printf("CUDA initialized on: %s\n", name);
}

CUcontext cuda_get_context(void) {
    if (!g_cuda_initialized) {
        cuda_driver_init();
    }
    return g_cuda_context;
}
