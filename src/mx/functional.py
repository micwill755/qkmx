import tensor_c

def softmax(input, dim=-1):
    """Apply softmax along dimension"""
    return input.softmax(dim)

def gelu(input):
    """Apply GELU activation"""
    return input.gelu()

def triu(size, diagonal=0):
    """Upper triangular matrix"""
    return tensor_c.triu(size, diagonal)

def randn(shape):
    """Random normal tensor"""
    return tensor_c.randn(shape)

def zeros(shape):
    return tensor_c.zeros(shape)

def ones(shape):
    return tensor_c.ones(shape)