import tensor_c
from .tensor import Tensor, float32

def softmax(input, dim=-1):
    """Apply softmax along dimension"""
    return input.softmax(dim)

def gelu(input):
    """Apply GELU activation"""
    return input.gelu()

def triu(size, diagonal=0):
    """Upper triangular matrix"""
    c_tensor = tensor_c.triu(size, diagonal)
    return Tensor._from_c_tensor(c_tensor)

def randn(shape, dtype=float32, device='cpu'):
    """Random normal tensor"""
    c_tensor = tensor_c.randn(list(shape), dtype, device=device)
    return Tensor._from_c_tensor(c_tensor)

def zeros(shape, dtype=float32, device='cpu'):
    """Tensor filled with zeros"""
    c_tensor = tensor_c.zeros(list(shape), dtype, device=device)
    return Tensor._from_c_tensor(c_tensor)

def ones(shape, dtype=float32, device='cpu'):
    """Tensor filled with ones"""
    c_tensor = tensor_c.ones(list(shape), dtype, device=device)
    return Tensor._from_c_tensor(c_tensor)

def array(data, device='cpu'):
    """Convert Python list or NumPy array to mx.Tensor"""
    import numpy as np
    if isinstance(data, np.ndarray):
        c_tensor = tensor_c.from_numpy(data, float32, device=device)
    else:
        c_tensor = tensor_c.from_list(data, float32, device=device)
    return Tensor._from_c_tensor(c_tensor)

def mean(input):
    """Compute mean of all elements"""
    return input.mean()