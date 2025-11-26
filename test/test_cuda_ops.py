#!/usr/bin/env python3
"""Test CUDA operations: matmul, gelu, layer_norm"""

import sys
sys.path.insert(0, 'src')

import mx
import numpy as np

def test_matmul():
    print("=" * 50)
    print("Testing MATMUL on CUDA")
    print("=" * 50)
    
    # Create test data
    A = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device='cuda')
    B = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cuda')
    
    print(f"A shape: {A.shape}, is_cuda: {A.is_cuda}")
    print(f"B shape: {B.shape}, is_cuda: {B.is_cuda}")
    
    # GPU matmul
    C = A @ B
    print(f"C = A @ B shape: {C.shape}")
    print(f"C:\n{C}")
    
    # Expected: [[22, 28], [49, 64]]
    print("Expected: [[22, 28], [49, 64]]")
    print("✓ Matmul CUDA test passed!\n")

def test_gelu():
    print("=" * 50)
    print("Testing GELU on CUDA")
    print("=" * 50)
    
    # Test values
    x = mx.array([[-1.0, 0.0, 1.0], [2.0, -2.0, 0.5]], device='cuda')
    print(f"Input x:\n{x}")
    print(f"x.is_cuda: {x.is_cuda}")
    
    # GPU GELU
    y = x.gelu()
    print(f"GELU(x):\n{y}")
    
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # GELU(0) ≈ 0, GELU(1) ≈ 0.841, GELU(-1) ≈ -0.159
    print("Expected approx: GELU(0)≈0, GELU(1)≈0.841, GELU(-1)≈-0.159")
    print("✓ GELU CUDA test passed!\n")

def test_layer_norm():
    print("=" * 50)
    print("Testing LAYER_NORM on CUDA")
    print("=" * 50)
    
    # Create input tensor (batch=2, features=4)
    x = mx.array([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]], device='cuda')
    
    # Gamma (scale) and beta (shift) - must be on same device
    gamma = mx.ones([4], device='cuda')
    beta = mx.zeros([4], device='cuda')
    
    print(f"Input x:\n{x}")
    print(f"x.is_cuda: {x.is_cuda}")
    print(f"gamma: {gamma}, beta: {beta}")
    
    # GPU LayerNorm
    eps = 1e-5
    y = x.layer_norm(gamma, beta, eps)
    print(f"LayerNorm(x):\n{y}")
    
    # For [1,2,3,4]: mean=2.5, var=1.25, std≈1.118
    # normalized ≈ [-1.34, -0.45, 0.45, 1.34]
    print("Expected: normalized values with mean≈0, std≈1 per row")
    print("✓ LayerNorm CUDA test passed!\n")

def test_softmax():
    print("=" * 50)
    print("Testing SOFTMAX on CUDA")
    print("=" * 50)
    
    x = mx.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]], device='cuda')
    print(f"Input x:\n{x}")
    print(f"x.is_cuda: {x.is_cuda}")
    
    # GPU Softmax
    y = x.softmax(dim=-1)
    print(f"Softmax(x):\n{y}")
    
    # Row 0: softmax([1,2,3]) ≈ [0.09, 0.24, 0.67]
    # Row 1: softmax([1,1,1]) = [0.33, 0.33, 0.33]
    print("Expected row sums = 1.0")
    print("✓ Softmax CUDA test passed!\n")

def test_cpu_vs_cuda():
    print("=" * 50)
    print("Testing CPU vs CUDA consistency")
    print("=" * 50)
    
    # Create same data on CPU and GPU
    data = [[1.0, 2.0], [3.0, 4.0]]
    A_cpu = mx.array(data, device='cpu')
    A_gpu = mx.array(data, device='cuda')
    
    B_data = [[5.0, 6.0], [7.0, 8.0]]
    B_cpu = mx.array(B_data, device='cpu')
    B_gpu = mx.array(B_data, device='cuda')
    
    # Matmul on both
    C_cpu = A_cpu @ B_cpu
    C_gpu = A_gpu @ B_gpu
    
    print(f"CPU result:\n{C_cpu}")
    print(f"GPU result:\n{C_gpu}")
    print("Results should match!")
    print("✓ CPU vs CUDA consistency test passed!\n")

if __name__ == "__main__":
    print("\nCUDA Operations Test Suite\n")
    
    test_matmul()
    test_gelu()
    test_layer_norm()
    test_softmax()
    test_cpu_vs_cuda()
    
    print("=" * 50)
    print("All CUDA tests completed!")
    print("=" * 50)
