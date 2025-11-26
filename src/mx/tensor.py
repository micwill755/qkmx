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
            shape: tuple/list of dimensions, or numpy array/list data
            dtype: data type (float32, int8, etc.)
        """
        import numpy as np
        
        # Handle numpy array
        if isinstance(shape, np.ndarray):
            self._c_tensor = tensor_c.from_numpy(shape, dtype)
        # Handle nested list (actual data) like [[1,2],[3,4]]
        elif isinstance(shape, list) and len(shape) > 0 and not isinstance(shape[0], int):
            self._c_tensor = tensor_c.from_list(shape, dtype)
        # Handle shape tuple like (3, 4)
        elif isinstance(shape, (list, tuple)):
            self._c_tensor = tensor_c.Tensor(list(shape), dtype)
        else:
            raise TypeError("shape must be a list, tuple, or numpy array")

    
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
    
    def __truediv__(self, scalar):
        """Scalar division (Tensor / scalar)"""
        if not isinstance(scalar, (int, float)):
            raise TypeError("Can only divide Tensor by scalar")
        result_c = self._c_tensor.scalar_div(float(scalar))
        return Tensor._from_c_tensor(result_c)

    def __mul__(self, other):
        """Element-wise multiplication or scalar multiplication"""
        if isinstance(other, (int, float)):
            result_c = self._c_tensor.scalar_mul(float(other))
            return Tensor._from_c_tensor(result_c)
        if not isinstance(other, Tensor):
            raise TypeError("Can only multiply Tensor with Tensor or scalar")
        result_c = self._c_tensor * other._c_tensor
        return Tensor._from_c_tensor(result_c)

    def __rmul__(self, other):
        """Reverse multiplication (scalar * Tensor)"""
        return self.__mul__(other)

    def __sub__(self, other):
        """Element-wise subtraction"""
        if not isinstance(other, Tensor):
            raise TypeError("Can only subtract Tensor from Tensor")
        result_c = self._c_tensor - other._c_tensor
        return Tensor._from_c_tensor(result_c)

    def __pow__(self, exponent):
        """Element-wise power"""
        if not isinstance(exponent, (int, float)):
            raise TypeError("Exponent must be a scalar")
        return self * self if exponent == 2 else NotImplemented

    def __matmul__(self, other):
        """Matrix multiplication using @ operator (runs in C)"""
        if not isinstance(other, Tensor):
            raise TypeError("Can only matmul Tensor with Tensor")
        result_c = self._c_tensor.matmul(other._c_tensor)
        return Tensor._from_c_tensor(result_c)

    def matmul(self, other):
        """Matrix multiplication (runs in C)"""
        if not isinstance(other, Tensor):
            raise TypeError("Can only matmul Tensor with Tensor")
        result_c = self._c_tensor.matmul(other._c_tensor)
        return Tensor._from_c_tensor(result_c)
    
    def reshape(self, new_shape):
        """Reshape tensor (runs in C)"""
        result_c = self._c_tensor.reshape(list(new_shape))
        return Tensor._from_c_tensor(result_c)
    
    def layer_norm(self, gamma, beta, eps):
        """Layer normalization (runs in C)"""
        result_c = self._c_tensor.layer_norm(gamma._c_tensor, beta._c_tensor, eps)
        return Tensor._from_c_tensor(result_c)

    def rms_norm(self, weight, eps):
        """RMS normalization (runs in C)"""
        result_c = self._c_tensor.rms_norm(weight._c_tensor, eps)
        return Tensor._from_c_tensor(result_c)

    def transpose(self, dim0, dim1):
        """Transpose two dimensions (runs in C)"""
        result_c = self._c_tensor.transpose(dim0, dim1)
        return Tensor._from_c_tensor(result_c)
    
    def softmax(self, dim=-1):
        """Softmax activation (runs in C)"""
        result_c = self._c_tensor.softmax(dim)
        return Tensor._from_c_tensor(result_c)

    def gelu(self):
        """GELU activation (runs in C)"""
        result_c = self._c_tensor.gelu()
        return Tensor._from_c_tensor(result_c)

    def mean(self):
        """Compute mean of all elements (runs in C)"""
        return self._c_tensor.mean()

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
    
    @property
    def is_cuda(self):
        """True if tensor is on CUDA device"""
        return self._c_tensor.is_cuda
    
    def cuda(self):
        """Move tensor to CUDA device"""
        result_c = self._c_tensor.cuda()
        return Tensor._from_c_tensor(result_c)
    
    def cpu(self):
        """Move tensor to CPU"""
        result_c = self._c_tensor.cpu()
        return Tensor._from_c_tensor(result_c)
    
    def __str__(self):
        """Delegate to C tensor's string representation"""
        return str(self._c_tensor)

    def __repr__(self):
        """Delegate to C tensor's string representation"""
        return str(self._c_tensor)
    
    def __getitem__(self, index):
        """Get item by index"""
        # If index is an mx.Tensor, extract the C tensor
        if isinstance(index, Tensor):
            result = self._c_tensor[index._c_tensor]
        else:
            result = self._c_tensor[index]
        
        if hasattr(result, 'shape'):
            return Tensor._from_c_tensor(result)
        return result
    
    def __setitem__(self, index, value):
        """Set item by index"""
        if isinstance(value, Tensor):
            self._c_tensor[index] = value._c_tensor
        else:
            self._c_tensor[index] = value

def zeros(shape, dtype=float32, device='cpu'):
    """Create tensor filled with zeros"""
    c_tensor = tensor_c.zeros(list(shape), dtype, device=device)
    return Tensor._from_c_tensor(c_tensor)

def ones(shape, dtype=float32, device='cpu'):
    """Create tensor filled with ones"""
    c_tensor = tensor_c.ones(list(shape), dtype, device=device)
    return Tensor._from_c_tensor(c_tensor)

def randn(shape, dtype=float32, device='cpu'):
    """Create tensor with random normal values"""
    c_tensor = tensor_c.randn(list(shape), dtype, device=device)
    return Tensor._from_c_tensor(c_tensor)

def rand(shape, dtype=float32, device='cpu'):
    """Create tensor with random uniform [0, 1)"""
    c_tensor = tensor_c.randn(list(shape), dtype, device=device)  # TODO: add rand to C
    return Tensor._from_c_tensor(c_tensor)

def empty(shape, dtype=float32, device='cpu'):
    """Create uninitialized tensor (fastest)"""
    c_tensor = tensor_c.zeros(list(shape), dtype, device=device)
    return Tensor._from_c_tensor(c_tensor)

def zeros_like(tensor):
    device = 'cuda' if tensor.is_cuda else 'cpu'
    return zeros(tensor.shape, device=device)

def from_list(data, dtype=float32, device='cpu'):
    """Create tensor from nested Python list"""
    c_tensor = tensor_c.from_list(data, dtype, device=device)
    return Tensor._from_c_tensor(c_tensor)

def from_numpy(array, dtype=float32, device='cpu'):
    """Create tensor from numpy array"""
    c_tensor = tensor_c.from_numpy(array, dtype, device=device)
    return Tensor._from_c_tensor(c_tensor)