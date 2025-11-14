# src/mx/tensor/tensor.py
"""
PyTorch-style tensor with pure C backend.
All operations delegated to C for maximum performance.
"""
import tensor_c

# Dtype constants
float32 = tensor_c.FLOAT32
float16 = tensor_c.FLOAT16
int8 = tensor_c.INT8
int4 = tensor_c.INT4
uint8 = tensor_c.UINT8

class Tensor:
    """Tensor with C backend """
    
    def __init__(self, shape, dtype=float32):
        """
        Create a new tensor.
        
        Args:
            shape: tuple or list of dimensions, e.g., (3, 4, 5)
            dtype: data type (float32, int8, etc.)
        """
        if isinstance(shape, (list, tuple)):
            self._c_tensor = tensor_c.Tensor(list(shape), dtype)
        else:
            raise TypeError("shape must be a list or tuple")
    
    @classmethod
    def _from_c_tensor(cls, c_tensor):
        """Internal: wrap existing C tensor"""
        obj = cls.__new__(cls)
        obj._c_tensor = c_tensor
        return obj
    
    def __add__(self, other):
        """Element-wise addition (runs in C)"""
        if not isinstance(other, Tensor):
            raise TypeError("Can only add Tensor to Tensor")
        result_c = self._c_tensor + other._c_tensor
        return Tensor._from_c_tensor(result_c)
    
    def __mul__(self, other):
        """Element-wise multiplication (runs in C)"""
        if not isinstance(other, Tensor):
            raise TypeError("Can only multiply Tensor with Tensor")
        result_c = self._c_tensor * other._c_tensor
        return Tensor._from_c_tensor(result_c)
    
    def __sub__(self, other):
        """Element-wise subtraction"""
        # TODO: implement in C
        raise NotImplementedError("Subtraction not yet implemented in C backend")
    
    def matmul(self, other):
        """Matrix multiplication (runs in C)"""
        if not isinstance(other, Tensor):
            raise TypeError("Can only matmul Tensor with Tensor")
        result_c = self._c_tensor.matmul(other._c_tensor)
        return Tensor._from_c_tensor(result_c)
    
    @property
    def shape(self):
        """Get tensor shape"""
        return tuple(self._c_tensor.shape)
    
    @property
    def dtype(self):
        """Get tensor data type"""
        return self._c_tensor.dtype
    
    @property
    def ndim(self):
        """Number of dimensions"""
        return len(self.shape)
    
    @property
    def size(self):
        """Total number of elements"""
        s = 1
        for d in self.shape:
            s *= d
        return s
    
    def __str__(self):
        """Delegate to C tensor's string representation"""
        return str(self._c_tensor)

    def __repr__(self):
        """Delegate to C tensor's string representation"""
        return str(self._c_tensor)

def zeros(shape, dtype=float32):
    """Create tensor filled with zeros"""
    return Tensor(shape, dtype)  # Already zeros from calloc!

def ones(shape, dtype=float32):
    """Create tensor filled with ones"""
    t = Tensor(shape, dtype)
    # TODO: call C function to fill with 1.0
    return t

def randn(shape, dtype=float32):
    """Create tensor with random normal values"""
    c_tensor = tensor_c.randn(list(shape), dtype)
    return Tensor._from_c_tensor(c_tensor)

def rand(shape, dtype=float32):
    """Create tensor with random uniform [0, 1)"""
    c_tensor = tensor_c.rand(list(shape), dtype)
    return Tensor._from_c_tensor(c_tensor)

def empty(shape, dtype=float32):
    """Create uninitialized tensor (fastest)"""
    return Tensor(shape, dtype)