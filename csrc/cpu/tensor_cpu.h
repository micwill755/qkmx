// tensor_cpu.h - CPU tensor memory operations
#ifndef TENSOR_CPU_H
#define TENSOR_CPU_H

#include "../tensor.h"

// CPU memory allocation
Tensor* tensor_create_cpu(int* shape, int ndim, DType dtype);
void tensor_free_cpu(Tensor* t);

// CPU data operations
void tensor_fill_cpu(Tensor* t, float value);
void tensor_copy_cpu(Tensor* dst, const Tensor* src);

#endif // TENSOR_CPU_H
