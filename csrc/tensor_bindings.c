// tensor_bindings.c - Python bindings using qkbind
#include "qkbind.h"
#include "tensor.h"
#include <string.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#ifdef USE_CUDA
#include <cuda.h>
#endif

// Forward declare the type
static PyTypeObject PyTensorType;

// Wrap Tensor struct
QKBIND_WRAP(Tensor, Tensor)

// __init__ - Create tensor from shape and dtype
QKBIND_INIT(Tensor, Tensor, tensor_create(shape, ndim, dtype, DEVICE_CPU),
    PyObject* shape_obj;
    int dtype = DTYPE_FLOAT32;
    
    if (!PyArg_ParseTuple(args, "O|i", &shape_obj, &dtype)) return -1;
    
    if (!PyList_Check(shape_obj)) {
        PyErr_SetString(PyExc_TypeError, "shape must be a list");
        return -1;
    }
    
    int ndim = PyList_Size(shape_obj);
    int* shape = (int*)malloc(ndim * sizeof(int));
    for (int i = 0; i < ndim; i++) {
        shape[i] = PyLong_AsLong(PyList_GetItem(shape_obj, i));
    }
)

// __del__ - Free tensor
QKBIND_DEALLOC(Tensor, tensor_free)

// Helper wrappers for tensor_op
static Tensor* tensor_add_wrapper(Tensor* a, Tensor* b) {
    return tensor_op(a, b, TENSOR_ADD);
}

static Tensor* tensor_mul_wrapper(Tensor* a, Tensor* b) {
    return tensor_op(a, b, TENSOR_MUL);
}

static Tensor* tensor_sub_wrapper(Tensor* a, Tensor* b) {
    return tensor_op(a, b, TENSOR_SUB);
}

QKBIND_BINOP(Tensor, add, tensor_add_wrapper)
QKBIND_BINOP(Tensor, mul, tensor_mul_wrapper)
QKBIND_BINOP(Tensor, sub, tensor_sub_wrapper)


QKBIND_METHOD(Tensor, matmul, tensor_matmul)

static PyObject* py_zeros(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* shape_obj;
    int dtype = DTYPE_FLOAT32;
    const char* device_str = "cpu";
    
    static char* kwlist[] = {"shape", "dtype", "device", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|is", kwlist, 
                                      &shape_obj, &dtype, &device_str)) return NULL;
    
    int ndim = PyList_Size(shape_obj);
    int* shape = (int*)malloc(ndim * sizeof(int));
    for (int i = 0; i < ndim; i++) {
        shape[i] = PyLong_AsLong(PyList_GetItem(shape_obj, i));
    }
    
    DeviceType device = (strcmp(device_str, "cuda") == 0) ? DEVICE_CUDA : DEVICE_CPU;
    Tensor* t = tensor_zeros(shape, ndim, dtype, device);
    free(shape);
    
    if (!t) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create tensor");
        return NULL;
    }
    
    PyTensorObject* result = PyObject_New(PyTensorObject, &PyTensorType);
    result->obj = t;
    return (PyObject*)result;
}

static PyObject* py_ones(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* shape_obj;
    int dtype = DTYPE_FLOAT32;
    const char* device_str = "cpu";
    
    static char* kwlist[] = {"shape", "dtype", "device", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|is", kwlist,
                                      &shape_obj, &dtype, &device_str)) return NULL;
    
    int ndim = PyList_Size(shape_obj);
    int* shape = (int*)malloc(ndim * sizeof(int));
    for (int i = 0; i < ndim; i++) {
        shape[i] = PyLong_AsLong(PyList_GetItem(shape_obj, i));
    }
    
    DeviceType device = (strcmp(device_str, "cuda") == 0) ? DEVICE_CUDA : DEVICE_CPU;
    Tensor* t = tensor_ones(shape, ndim, dtype, device);
    free(shape);
    
    if (!t) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create tensor");
        return NULL;
    }
    
    PyTensorObject* result = PyObject_New(PyTensorObject, &PyTensorType);
    result->obj = t;
    return (PyObject*)result;
}

// transpose method with optional args
static PyObject* PyTensor_transpose(PyTensorObject* self, PyObject* args) {
    int dim0 = 0, dim1 = 1;  // Default: swap first two dimensions
    
    // "|ii" means: both integers are optional
    if (!PyArg_ParseTuple(args, "|ii", &dim0, &dim1)) return NULL;
    
    // Create new Python tensor object
    PyTensorObject* result = PyObject_New(PyTensorObject, &PyTensorType);
    
    // Call your C function with the two dimension arguments
    result->obj = tensor_transpose(self->obj, dim0, dim1);
    
    // Check if it failed
    if (!result->obj) {
        Py_DECREF(result);
        PyErr_SetString(PyExc_ValueError, "Transpose failed");
        return NULL;
    }
    
    return (PyObject*)result;
}

// layer_norm method
static PyObject* PyTensor_layer_norm(PyTensorObject* self, PyObject* args) {
    PyTensorObject* gamma;
    PyTensorObject* beta;
    float eps = 1e-5;  // Default epsilon
    
    // Parse arguments: gamma, beta, optional eps
    if (!PyArg_ParseTuple(args, "O!O!|f", 
                          &PyTensorType, &gamma,
                          &PyTensorType, &beta,
                          &eps)) {
        return NULL;
    }
    
    // Create result tensor
    PyTensorObject* result = PyObject_New(PyTensorObject, &PyTensorType);
    result->obj = tensor_layer_norm(self->obj, gamma->obj, beta->obj, eps);
    
    if (!result->obj) {
        Py_DECREF(result);
        PyErr_SetString(PyExc_ValueError, "LayerNorm failed");
        return NULL;
    }
    
    return (PyObject*)result;
}

// rms_norm method
static PyObject* PyTensor_rms_norm(PyTensorObject* self, PyObject* args) {
    PyTensorObject* weight;
    float eps = 1e-5;
    
    if (!PyArg_ParseTuple(args, "O!|f", 
                          &PyTensorType, &weight,
                          &eps)) {
        return NULL;
    }
    
    PyTensorObject* result = PyObject_New(PyTensorObject, &PyTensorType);
    result->obj = tensor_rms_norm(self->obj, weight->obj, eps);
    
    if (!result->obj) {
        Py_DECREF(result);
        PyErr_SetString(PyExc_ValueError, "RMSNorm failed");
        return NULL;
    }
    
    return (PyObject*)result;
}

// Properties
QKBIND_PROPERTY_INT_ARRAY(Tensor, shape, shape, ndim)
QKBIND_PROPERTY_INT(Tensor, dtype, dtype)
QKBIND_PROPERTY_SIZE(Tensor, size, size)
QKBIND_PROPERTY_INT(Tensor, ndim, ndim)


static PyObject* PyTensor_get_T(PyTensorObject* self, void* closure) {
    PyTensorObject* result = PyObject_New(PyTensorObject, &PyTensorType);
    result->obj = tensor_transpose(self->obj, 0, 1);  // Default transpose
    if (!result->obj) {
        Py_DECREF(result);
        PyErr_SetString(PyExc_ValueError, "Transpose failed");
        return NULL;
    }
    return (PyObject*)result;
}

// reshape method
static PyObject* PyTensor_reshape(PyTensorObject* self, PyObject* args) {
    PyObject* shape_obj;
    
    if (!PyArg_ParseTuple(args, "O", &shape_obj)) return NULL;
    
    if (!PyList_Check(shape_obj) && !PyTuple_Check(shape_obj)) {
        PyErr_SetString(PyExc_TypeError, "shape must be a list or tuple");
        return NULL;
    }
    
    int new_ndim = PySequence_Size(shape_obj);
    int* new_shape = (int*)malloc(new_ndim * sizeof(int));
    for (int i = 0; i < new_ndim; i++) {
        PyObject* item = PySequence_GetItem(shape_obj, i);
        new_shape[i] = PyLong_AsLong(item);
        Py_DECREF(item);
    }
    
    PyTensorObject* result = PyObject_New(PyTensorObject, &PyTensorType);
    result->obj = tensor_reshape(self->obj, new_shape, new_ndim);
    free(new_shape);
    
    if (!result->obj) {
        Py_DECREF(result);
        PyErr_SetString(PyExc_ValueError, "Reshape failed - size mismatch");
        return NULL;
    }
    
    return (PyObject*)result;
}

// triu - module level function
static PyObject* py_triu(PyObject* self, PyObject* args) {
    int size, diagonal = 0;
    
    if (!PyArg_ParseTuple(args, "i|i", &size, &diagonal)) return NULL;
    
    PyTensorObject* result = PyObject_New(PyTensorObject, &PyTensorType);
    result->obj = tensor_triu(size, diagonal);
    
    if (!result->obj) {
        Py_DECREF(result);
        PyErr_SetString(PyExc_ValueError, "triu failed");
        return NULL;
    }
    
    return (PyObject*)result;
}

// masked_fill - tensor method
static PyObject* PyTensor_masked_fill(PyTensorObject* self, PyObject* args) {
    PyTensorObject* mask;
    float value;
    
    if (!PyArg_ParseTuple(args, "O!f", &PyTensorType, &mask, &value)) return NULL;
    
    PyTensorObject* result = PyObject_New(PyTensorObject, &PyTensorType);
    result->obj = tensor_masked_fill(self->obj, mask->obj, value);
    
    if (!result->obj) {
        Py_DECREF(result);
        PyErr_SetString(PyExc_ValueError, "masked_fill failed");
        return NULL;
    }
    
    return (PyObject*)result;
}


// Get scalar value method
static PyObject* PyTensor_item(PyTensorObject* self, PyObject* args) {
    // Parse indices
    PyObject* indices_obj = NULL;
    if (!PyArg_ParseTuple(args, "|O", &indices_obj)) return NULL;
    
    if (indices_obj == NULL || indices_obj == Py_None) {
        // No indices - return first element
        if (self->obj->size != 1) {
            PyErr_SetString(PyExc_ValueError, "Can only convert 1-element tensor to scalar");
            return NULL;
        }
        float val = ((float*)self->obj->data)[0];
        return PyFloat_FromDouble(val);
    }
    
    // TODO: Handle tuple of indices
    PyErr_SetString(PyExc_NotImplementedError, "Indexed item() not yet implemented");
    return NULL;
}

// scalar_div method
static PyObject* PyTensor_scalar_div(PyTensorObject* self, PyObject* args) {
    float scalar;
    
    if (!PyArg_ParseTuple(args, "f", &scalar)) {
        return NULL;
    }
    
    if (scalar == 0.0f) {
        PyErr_SetString(PyExc_ZeroDivisionError, "Division by zero");
        return NULL;
    }
    
    PyTensorObject* result = PyObject_New(PyTensorObject, &PyTensorType);
    result->obj = tensor_scalar_op(self->obj, scalar, TENSOR_DIV);
    
    if (!result->obj) {
        Py_DECREF(result);
        PyErr_SetString(PyExc_ValueError, "Scalar division failed");
        return NULL;
    }
    
    return (PyObject*)result;
}

// scalar_mul method
static PyObject* PyTensor_scalar_mul(PyTensorObject* self, PyObject* args) {
    float scalar;
    
    if (!PyArg_ParseTuple(args, "f", &scalar)) {
        return NULL;
    }
    
    PyTensorObject* result = PyObject_New(PyTensorObject, &PyTensorType);
    result->obj = tensor_scalar_op(self->obj, scalar, TENSOR_MUL);
    
    if (!result->obj) {
        Py_DECREF(result);
        PyErr_SetString(PyExc_ValueError, "Scalar multiplication failed");
        return NULL;
    }
    
    return (PyObject*)result;
}

// softmax method
static PyObject* PyTensor_softmax(PyTensorObject* self, PyObject* args) {
    int dim = -1;
    if (!PyArg_ParseTuple(args, "|i", &dim)) return NULL;
    
    PyTensorObject* result = PyObject_New(PyTensorObject, &PyTensorType);
    result->obj = tensor_softmax(self->obj, dim);
    
    if (!result->obj) {
        Py_DECREF(result);
        PyErr_SetString(PyExc_ValueError, "Softmax failed");
        return NULL;
    }
    
    return (PyObject*)result;
}

// gelu method
static PyObject* PyTensor_gelu(PyTensorObject* self, PyObject* args) {
    PyTensorObject* result = PyObject_New(PyTensorObject, &PyTensorType);
    result->obj = tensor_gelu(self->obj);
    
    if (!result->obj) {
        Py_DECREF(result);
        PyErr_SetString(PyExc_ValueError, "GELU failed");
        return NULL;
    }
    
    return (PyObject*)result;
}

// mean method
static PyObject* PyTensor_mean(PyTensorObject* self, PyObject* args) {
    float result = tensor_mean(self->obj);
    return PyFloat_FromDouble(result);
}

// cuda method - move tensor to GPU
static PyObject* PyTensor_cuda(PyTensorObject* self, PyObject* args) {
    Tensor* result = tensor_to_cuda(self->obj);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to move tensor to CUDA (CUDA not available?)");
        return NULL;
    }
    
    PyTensorObject* py_result = PyObject_New(PyTensorObject, &PyTensorType);
    py_result->obj = result;
    return (PyObject*)py_result;
}

// cpu method - move tensor to CPU
static PyObject* PyTensor_cpu(PyTensorObject* self, PyObject* args) {
    Tensor* result = tensor_to_cpu(self->obj);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to move tensor to CPU");
        return NULL;
    }
    
    PyTensorObject* py_result = PyObject_New(PyTensorObject, &PyTensorType);
    py_result->obj = result;
    return (PyObject*)py_result;
}

// is_cuda property getter
static PyObject* PyTensor_get_is_cuda(PyTensorObject* self, void* closure) {
    return PyBool_FromLong(self->obj->device == DEVICE_CUDA);
}

// Method table
static PyMethodDef PyTensor_methods[] = {
    {"matmul", (PyCFunction)PyTensor_matmul, METH_VARARGS, "Matrix multiplication"},
    {"scalar_div", (PyCFunction)PyTensor_scalar_div, METH_VARARGS, "Scalar division"},
    {"scalar_mul", (PyCFunction)PyTensor_scalar_mul, METH_VARARGS, "Scalar multiplication"},
    {"transpose", (PyCFunction)PyTensor_transpose, METH_VARARGS, "Transpose tensor"},
    {"layer_norm", (PyCFunction)PyTensor_layer_norm, METH_VARARGS, "Layer normalization"},
    {"rms_norm", (PyCFunction)PyTensor_rms_norm, METH_VARARGS, "RMS normalization"},
    {"reshape", (PyCFunction)PyTensor_reshape, METH_VARARGS, "Reshape tensor"},
    {"masked_fill", (PyCFunction)PyTensor_masked_fill, METH_VARARGS, "Fill values where mask is true"},
    {"softmax", (PyCFunction)PyTensor_softmax, METH_VARARGS, "Softmax activation"},
    {"gelu", (PyCFunction)PyTensor_gelu, METH_VARARGS, "GELU activation"},
    {"mean", (PyCFunction)PyTensor_mean, METH_VARARGS, "Compute mean"},
    {"item", (PyCFunction)PyTensor_item, METH_VARARGS, "Get scalar value"},
    {"cuda", (PyCFunction)PyTensor_cuda, METH_NOARGS, "Move tensor to CUDA"},
    {"cpu", (PyCFunction)PyTensor_cpu, METH_NOARGS, "Move tensor to CPU"},
    {NULL} 
};

// Property table
static PyGetSetDef PyTensor_getset[] = {
    {"shape", (getter)PyTensor_get_shape, NULL, "Tensor shape", NULL},
    {"dtype", (getter)PyTensor_get_dtype, NULL, "Data type", NULL},
    {"size", (getter)PyTensor_get_size, NULL, "Total elements", NULL},
    {"ndim", (getter)PyTensor_get_ndim, NULL, "Number of dimensions", NULL},
    {"T", (getter)PyTensor_get_T, NULL, "Transposed tensor", NULL},
    {"is_cuda", (getter)PyTensor_get_is_cuda, NULL, "True if tensor is on CUDA", NULL},
    {NULL}
};

static PyNumberMethods PyTensor_as_number = {
    .nb_add = (binaryfunc)PyTensor_add,
    .nb_subtract = (binaryfunc)PyTensor_sub,
    .nb_multiply = (binaryfunc)PyTensor_mul,
    .nb_matrix_multiply = (binaryfunc)PyTensor_matmul,
};

// Helper function for building string representation
static void build_str(char** ptr, int* shape, int ndim, float* data, size_t* offset, int indent) {
    if (ndim == 1) {
        *ptr += sprintf(*ptr, "[");
        for (int i = 0; i < shape[0]; i++) {
            if (i > 0) *ptr += sprintf(*ptr, ", ");
            *ptr += sprintf(*ptr, "%.4f", data[(*offset)++]);
        }
        *ptr += sprintf(*ptr, "]");
        return;
    }
    
    *ptr += sprintf(*ptr, "[");
    for (int i = 0; i < shape[0]; i++) {
        if (i > 0) {
            *ptr += sprintf(*ptr, ",\n%*s", indent + 1, "");
        }
        build_str(ptr, shape + 1, ndim - 1, data, offset, indent + 1);
    }
    *ptr += sprintf(*ptr, "]");
}

// Format tensor as string (PyTorch style)
static PyObject* PyTensor_str(PyTensorObject* self) {
    Tensor* t = self->obj;
    
    if (t->dtype != DTYPE_FLOAT32) {
        return PyUnicode_FromFormat("tensor(shape=%R, dtype=%d, device=%s)", 
                                    PyTensor_get_shape(self, NULL), t->dtype,
                                    t->device == DEVICE_CUDA ? "cuda" : "cpu");
    }
    
    // If tensor is on GPU, copy to CPU for printing
    float* data;
    float* temp_buffer = NULL;
    
    if (t->device == DEVICE_CUDA) {
#ifdef USE_CUDA
        temp_buffer = (float*)malloc(t->size * sizeof(float));
        if (!temp_buffer) {
            return PyUnicode_FromString("tensor(...)");
        }
        // Copy from GPU to CPU
        extern void cuda_driver_init(void);
        cuda_driver_init();
        cuMemcpyDtoH(temp_buffer, (CUdeviceptr)t->data, t->size * sizeof(float));
        data = temp_buffer;
#else
        return PyUnicode_FromString("tensor(cuda, no CUDA support)");
#endif
    } else {
        data = (float*)t->data;
    }
    
    static char buffer[10000];
    char* ptr = buffer;
    
    size_t offset = 0;
    build_str(&ptr, t->shape, t->ndim, data, &offset, 0);
    
    if (temp_buffer) {
        free(temp_buffer);
    }
    
    const char* device_str = (t->device == DEVICE_CUDA) ? ", device='cuda'" : "";
    return PyUnicode_FromFormat("tensor(%s%s)", buffer, device_str);
}

// Module-level function for randn
static PyObject* py_randn(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* shape_obj;
    int dtype = DTYPE_FLOAT32;
    const char* device_str = "cpu";
    
    static char* kwlist[] = {"shape", "dtype", "device", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|is", kwlist,
                                      &shape_obj, &dtype, &device_str)) return NULL;
    
    int ndim = PyList_Size(shape_obj);
    int* shape = (int*)malloc(ndim * sizeof(int));
    for (int i = 0; i < ndim; i++) {
        shape[i] = PyLong_AsLong(PyList_GetItem(shape_obj, i));
    }
    
    DeviceType device = (strcmp(device_str, "cuda") == 0) ? DEVICE_CUDA : DEVICE_CPU;
    Tensor* t = tensor_randn(shape, ndim, dtype, device);
    free(shape);
    
    if (!t) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create tensor");
        return NULL;
    }
    
    PyTensorObject* result = PyObject_New(PyTensorObject, &PyTensorType);
    result->obj = t;
    return (PyObject*)result;
}

// __getitem__ - tensor indexing
static PyObject* PyTensor_getitem(PyTensorObject* self, PyObject* key) {
    // Handle integer index
    if (PyLong_Check(key)) {
        long index = PyLong_AsLong(key);
        
        // Handle negative indexing
        if (index < 0) {
            index += self->obj->shape[0];
        }
        
        // Bounds check
        if (index < 0 || index >= self->obj->shape[0]) {
            PyErr_SetString(PyExc_IndexError, "Index out of bounds");
            return NULL;
        }
        
        // Get slice at index
        Tensor* result = tensor_get_index(self->obj, index);
        if (!result) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to get index");
            return NULL;
        }
        
        PyTensorObject* py_result = PyObject_New(PyTensorObject, &PyTensorType);
        py_result->obj = result;
        return (PyObject*)py_result;
    }
    
    // Handle tensor index (advanced indexing)
    if (PyObject_TypeCheck(key, &PyTensorType)) {
        PyTensorObject* index_tensor = (PyTensorObject*)key;
        
        // Call C function for advanced indexing
        Tensor* result = tensor_advanced_index(self->obj, index_tensor->obj);
        if (!result) {
            PyErr_SetString(PyExc_RuntimeError, "Advanced indexing failed");
            return NULL;
        }
        
        PyTensorObject* py_result = PyObject_New(PyTensorObject, &PyTensorType);
        py_result->obj = result;
        return (PyObject*)py_result;
    }
    
    PyErr_SetString(PyExc_TypeError, "Index must be an integer or Tensor");
    return NULL;
}

// __setitem__ - tensor index assignment
static int PyTensor_setitem(PyTensorObject* self, PyObject* key, PyObject* value) {
    if (!PyLong_Check(key)) {
        PyErr_SetString(PyExc_TypeError, "Index must be an integer");
        return -1;
    }
    
    long index = PyLong_AsLong(key);
    
    // Handle negative indexing
    if (index < 0) {
        index += self->obj->shape[0];
    }
    
    // Bounds check
    if (index < 0 || index >= self->obj->shape[0]) {
        PyErr_SetString(PyExc_IndexError, "Index out of bounds");
        return -1;
    }
    
    // Handle scalar value
    if (PyFloat_Check(value) || PyLong_Check(value)) {
        float val = PyFloat_Check(value) ? PyFloat_AsDouble(value) : PyLong_AsDouble(value);
        if (tensor_set_scalar(self->obj, index, val) != 0) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to set scalar");
            return -1;
        }
        return 0;
    }
    
    // Handle tensor value
    if (PyObject_TypeCheck(value, &PyTensorType)) {
        PyTensorObject* tensor_val = (PyTensorObject*)value;
        if (tensor_set_index(self->obj, index, tensor_val->obj) != 0) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to set index");
            return -1;
        }
        return 0;
    }
    
    PyErr_SetString(PyExc_TypeError, "Value must be a number or Tensor");
    return -1;
}

// Add this function before module_methods array
static PyObject* py_from_numpy(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* np_array_obj;
    int dtype = DTYPE_FLOAT32;
    const char* device_str = "cpu";
    
    static char* kwlist[] = {"array", "dtype", "device", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|is", kwlist,
                                      &np_array_obj, &dtype, &device_str)) return NULL;
    
    if (!PyArray_Check(np_array_obj)) {
        PyErr_SetString(PyExc_TypeError, "Expected numpy array");
        return NULL;
    }
    
    // Ensure array is contiguous and in float32 (allow any casting)
    PyArrayObject* np_array = (PyArrayObject*)PyArray_FROM_OTF(np_array_obj, NPY_FLOAT32, 
                                                                NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    if (!np_array) {
        return NULL;  // PyArray_FROM_OTF already set the error
    }
    
    // Get shape
    int ndim = PyArray_NDIM(np_array);
    npy_intp* np_dims = PyArray_DIMS(np_array);
    int* shape = (int*)malloc(ndim * sizeof(int));
    for (int i = 0; i < ndim; i++) {
        shape[i] = (int)np_dims[i];
    }
    
    // Create tensor on CPU first (need to copy data)
    Tensor* t = tensor_create(shape, ndim, dtype, DEVICE_CPU);
    free(shape);
    
    if (!t) {
        Py_DECREF(np_array);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create tensor");
        return NULL;
    }
    
    // Copy data
    void* np_data = PyArray_DATA(np_array);
    size_t total = PyArray_SIZE(np_array);
    memcpy(t->data, np_data, total * sizeof(float));
    
    Py_DECREF(np_array);  // Release the converted array
    
    // Move to GPU if requested
    if (strcmp(device_str, "cuda") == 0) {
        Tensor* gpu_t = tensor_to_cuda(t);
        tensor_free(t);
        if (!gpu_t) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to move tensor to CUDA");
            return NULL;
        }
        t = gpu_t;
    }
    
    // Wrap and return
    PyTensorObject* result = PyObject_New(PyTensorObject, &PyTensorType);
    result->obj = t;
    
    return (PyObject*)result;
}

static PyMappingMethods PyTensor_as_mapping = {
    .mp_subscript = (binaryfunc)PyTensor_getitem,
    .mp_ass_subscript = (objobjargproc)PyTensor_setitem,
};

// Helper to recursively parse nested Python lists and fill tensor data
static int parse_nested_list(PyObject* list, float* data, int* shape, int ndim, int current_dim, size_t* offset) {
    if (current_dim == ndim - 1) {
        // Base case: innermost dimension
        if (!PyList_Check(list)) return -1;
        Py_ssize_t size = PyList_Size(list);
        if (size != shape[current_dim]) return -1;
        
        for (Py_ssize_t i = 0; i < size; i++) {
            PyObject* item = PyList_GetItem(list, i);
            if (PyFloat_Check(item)) {
                data[(*offset)++] = PyFloat_AsDouble(item);
            } else if (PyLong_Check(item)) {
                data[(*offset)++] = (float)PyLong_AsLong(item);
            } else {
                return -1;
            }
        }
        return 0;
    }
    
    // Recursive case
    if (!PyList_Check(list)) return -1;
    Py_ssize_t size = PyList_Size(list);
    if (size != shape[current_dim]) return -1;
    
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject* sublist = PyList_GetItem(list, i);
        if (parse_nested_list(sublist, data, shape, ndim, current_dim + 1, offset) != 0) {
            return -1;
        }
    }
    return 0;
}

// Helper to infer shape from nested Python list
static int infer_shape(PyObject* list, int* shape, int max_ndim, int current_dim) {
    if (!PyList_Check(list)) return current_dim;
    
    Py_ssize_t size = PyList_Size(list);
    if (size == 0) return current_dim;
    
    shape[current_dim] = size;
    
    PyObject* first = PyList_GetItem(list, 0);
    if (PyList_Check(first)) {
        return infer_shape(first, shape, max_ndim, current_dim + 1);
    }
    
    return current_dim + 1;
}

// Create tensor from nested Python list
static PyObject* py_from_list(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* list_obj;
    int dtype = DTYPE_FLOAT32;
    const char* device_str = "cpu";
    
    static char* kwlist[] = {"data", "dtype", "device", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|is", kwlist,
                                      &list_obj, &dtype, &device_str)) return NULL;
    
    if (!PyList_Check(list_obj)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list");
        return NULL;
    }
    
    // Infer shape
    int shape[8];  // Max 8 dimensions
    int ndim = infer_shape(list_obj, shape, 8, 0);
    
    if (ndim == 0) {
        PyErr_SetString(PyExc_ValueError, "Cannot create tensor from empty list");
        return NULL;
    }
    
    // Create tensor on CPU first (need to fill data)
    Tensor* t = tensor_create(shape, ndim, dtype, DEVICE_CPU);
    if (!t) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create tensor");
        return NULL;
    }
    
    // Fill data
    size_t offset = 0;
    if (parse_nested_list(list_obj, (float*)t->data, shape, ndim, 0, &offset) != 0) {
        tensor_free(t);
        PyErr_SetString(PyExc_ValueError, "Invalid nested list structure");
        return NULL;
    }
    
    // Move to GPU if requested
    if (strcmp(device_str, "cuda") == 0) {
        Tensor* gpu_t = tensor_to_cuda(t);
        tensor_free(t);
        if (!gpu_t) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to move tensor to CUDA");
            return NULL;
        }
        t = gpu_t;
    }
    
    // Wrap and return
    PyTensorObject* result = PyObject_New(PyTensorObject, &PyTensorType);
    result->obj = t;
    return (PyObject*)result;
}

// Module methods
static PyMethodDef module_methods[] = {
    {"randn", (PyCFunction)py_randn, METH_VARARGS | METH_KEYWORDS, "Random normal tensor"},
    {"zeros", (PyCFunction)py_zeros, METH_VARARGS | METH_KEYWORDS, "Tensor filled with zeros"},
    {"ones", (PyCFunction)py_ones, METH_VARARGS | METH_KEYWORDS, "Tensor filled with ones"},
    {"triu", py_triu, METH_VARARGS, "Upper triangular matrix"},
    {"from_numpy", (PyCFunction)py_from_numpy, METH_VARARGS | METH_KEYWORDS, "Create tensor from numpy array"},
    {"from_list", (PyCFunction)py_from_list, METH_VARARGS | METH_KEYWORDS, "Create tensor from nested Python list"},
    {NULL}
};

// Type definition
QKBIND_TYPE_BEGIN(Tensor, tensor_c)
    .tp_methods = PyTensor_methods,
    .tp_getset = PyTensor_getset,
    .tp_as_number = &PyTensor_as_number,
    .tp_as_mapping = &PyTensor_as_mapping,
    .tp_str = (reprfunc)PyTensor_str,
    .tp_doc = "Fast C Tensor",
QKBIND_TYPE_END

// Module definition
static PyModuleDef tensor_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "tensor_c",
    .m_size = -1,
    .m_methods = module_methods,
};

PyMODINIT_FUNC PyInit_tensor_c(void) {
    import_array();  // Initialize NumPy C API
    
    PyObject* m = PyModule_Create(&tensor_module);
    if (!m) return NULL;
    
    if (PyType_Ready(&PyTensorType) < 0) return NULL;
    
    Py_INCREF(&PyTensorType);
    PyModule_AddObject(m, "Tensor", (PyObject*)&PyTensorType);
    
    PyModule_AddIntConstant(m, "FLOAT32", DTYPE_FLOAT32);
    PyModule_AddIntConstant(m, "FLOAT16", DTYPE_FLOAT16);
    PyModule_AddIntConstant(m, "INT8", DTYPE_INT8);
    PyModule_AddIntConstant(m, "INT4", DTYPE_INT4);
    PyModule_AddIntConstant(m, "UINT8", DTYPE_UINT8);
    
    return m;
}
