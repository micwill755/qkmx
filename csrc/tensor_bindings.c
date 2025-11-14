// tensor_bindings.c - Python bindings using qkbind
#include "qkbind.h"
#include "tensor.h"

// Forward declare the type
static PyTypeObject PyTensorType;

// Wrap Tensor struct
QKBIND_WRAP(Tensor, Tensor)

// __init__ - Create tensor from shape and dtype
QKBIND_INIT(Tensor, Tensor, tensor_create(shape, ndim, dtype),
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

// __add__ operator
QKBIND_BINOP(Tensor, add, tensor_add_wrapper)

// __mul__ operator
QKBIND_BINOP(Tensor, mul, tensor_mul_wrapper)

// matmul method
QKBIND_METHOD(Tensor, matmul, tensor_matmul)

// Properties
QKBIND_PROPERTY_INT_ARRAY(Tensor, shape, shape, ndim)
QKBIND_PROPERTY_INT(Tensor, dtype, dtype)
QKBIND_PROPERTY_SIZE(Tensor, size, size)
QKBIND_PROPERTY_INT(Tensor, ndim, ndim)

// Method table
static PyMethodDef PyTensor_methods[] = {
    {"matmul", (PyCFunction)PyTensor_matmul, METH_VARARGS, "Matrix multiplication"},
    {NULL}
};

// Property table
static PyGetSetDef PyTensor_getset[] = {
    {"shape", (getter)PyTensor_get_shape, NULL, "Tensor shape", NULL},
    {"dtype", (getter)PyTensor_get_dtype, NULL, "Data type", NULL},
    {"size", (getter)PyTensor_get_size, NULL, "Total elements", NULL},
    {"ndim", (getter)PyTensor_get_ndim, NULL, "Number of dimensions", NULL},
    {NULL}
};

// Number methods (for operators)
static PyNumberMethods PyTensor_as_number = {
    .nb_add = (binaryfunc)PyTensor_add,
    .nb_multiply = (binaryfunc)PyTensor_mul,
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
        return PyUnicode_FromFormat("tensor(shape=%R, dtype=%d)", 
                                    PyTensor_get_shape(self, NULL), t->dtype);
    }
    
    float* data = (float*)t->data;
    static char buffer[10000];
    char* ptr = buffer;
    
    size_t offset = 0;
    build_str(&ptr, t->shape, t->ndim, data, &offset, 0);
    
    return PyUnicode_FromFormat("tensor(%s)", buffer);
}

// Module-level function for randn
static PyObject* py_randn(PyObject* self, PyObject* args) {
    PyObject* shape_obj;
    int dtype = DTYPE_FLOAT32;
    
    if (!PyArg_ParseTuple(args, "O|i", &shape_obj, &dtype)) return NULL;
    
    int ndim = PyList_Size(shape_obj);
    int* shape = (int*)malloc(ndim * sizeof(int));
    for (int i = 0; i < ndim; i++) {
        shape[i] = PyLong_AsLong(PyList_GetItem(shape_obj, i));
    }
    
    Tensor* t = tensor_randn(shape, ndim, dtype);
    free(shape);
    
    PyTensorObject* result = PyObject_New(PyTensorObject, &PyTensorType);
    result->obj = t;
    return (PyObject*)result;
}

// Module methods
static PyMethodDef module_methods[] = {
    {"randn", py_randn, METH_VARARGS, "Random normal tensor"},
    {NULL}
};

// Type definition
QKBIND_TYPE_BEGIN(Tensor, tensor_c)
    .tp_methods = PyTensor_methods,
    .tp_getset = PyTensor_getset,
    .tp_as_number = &PyTensor_as_number,
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
