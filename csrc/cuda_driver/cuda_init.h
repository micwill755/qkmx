// cuda_init.h - Shared CUDA Driver API initialization
#ifndef CUDA_INIT_H
#define CUDA_INIT_H

#include <cuda.h>

// Initialize CUDA Driver API (call once, safe to call multiple times)
void cuda_driver_init(void);

// Get the CUDA context
CUcontext cuda_get_context(void);

// Error checking macro
#define CU_CHECK(call) do { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char* errStr; \
        cuGetErrorString(err, &errStr); \
        fprintf(stderr, "CUDA Driver error at %s:%d: %s\n", __FILE__, __LINE__, errStr); \
        exit(1); \
    } \
} while(0)

#endif // CUDA_INIT_H
