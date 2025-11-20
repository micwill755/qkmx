import tensor_c
from .tensor import Tensor

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

def randn(shape):
    """Random normal tensor"""
    c_tensor = tensor_c.randn(list(shape))
    return Tensor._from_c_tensor(c_tensor)

def zeros(shape):
    return Tensor(shape)

def ones(shape):
    c_tensor = tensor_c.ones(list(shape))
    return Tensor._from_c_tensor(c_tensor)