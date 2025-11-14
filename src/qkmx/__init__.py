# src/qkmx/__init__.py
"""
qkmx - Fast tensor library with C backend
"""

from tensor.tensor import (  # Change this line
    Tensor,
    zeros,
    ones,
    randn,
    rand,
    empty,
    float32,
    float16,
    int8,
    int4,
    uint8,
)

__all__ = [
    'Tensor',
    'zeros',
    'ones',
    'randn',
    'rand',
    'empty',
    'float32',
    'float16',
    'int8',
    'int4',
    'uint8',
]
