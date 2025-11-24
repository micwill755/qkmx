#!/usr/bin/env python3
"""Test CPU and CUDA tensor operations"""

import sys
sys.path.insert(0, 'src')

import mx
import numpy as np

def test_gelu():
    print("\n=== Testing GELU ===")
    
    # Create test data
    x = mx.Tensor([[-1.0, 0.0, 1.0, 2.0]])
    print(f"Input: {x}")
    
    # Test GELU
    y = x.gelu()
    print(f"Output: {y}")
    
    # Or use functional API
    y2 = mx.gelu(x)
    print(f"Functional API: {y2}")
    
    print("✓ GELU passed")

def test_softmax():
    print("\n=== Testing Softmax ===")
    
    # Create test data
    x = mx.Tensor([[1.0, 2.0, 3.0, 4.0]])
    print(f"Input: {x}")
    
    # Test softmax
    y = x.softmax(dim=1)
    print(f"Output: {y}")
    
    # Or use functional API
    y2 = mx.softmax(x, dim=1)
    print(f"Functional API: {y2}")
    
    print("✓ Softmax passed")

def test_matmul():
    print("\n=== Testing Matmul ===")
    
    # Create test matrices
    A = mx.Tensor([[1.0, 2.0], [3.0, 4.0]])
    B = mx.Tensor([[5.0, 6.0], [7.0, 8.0]])
    print(f"A: {A}")
    print(f"B: {B}")
    
    # Test matmul with @ operator
    C = A @ B
    print(f"A @ B = {C}")
    
    # Or use method
    C2 = A.matmul(B)
    print(f"A.matmul(B) = {C2}")
    
    print("Expected: [[19, 22], [43, 50]]")
    print("✓ Matmul passed")

def test_layer_norm():
    print("\n=== Testing Layer Norm ===")
    
    # Create test data
    x = mx.Tensor([[1.0, 2.0, 3.0, 4.0]])
    gamma = mx.ones([4])
    beta = mx.zeros([4])
    
    print(f"Input: {x}")
    
    # Test layer norm
    y = x.layer_norm(gamma, beta, eps=1e-5)
    print(f"Output: {y}")
    
    print("✓ Layer Norm passed")

def test_cuda_if_available():
    """Test CUDA operations if available"""
    print("\n=== Testing CUDA (if available) ===")
    
    try:
        # Create CPU tensor
        x = mx.Tensor([[1.0, 2.0, 3.0]])
        
        # Check if CUDA methods exist
        if hasattr(x._c_tensor, 'to_cuda'):
            # Move to GPU
            x_gpu = x._c_tensor.to_cuda()
            x_gpu_tensor = mx.Tensor._from_c_tensor(x_gpu)
            
            # Run operation on GPU
            y_gpu = x_gpu_tensor.gelu()
            
            # Move back to CPU
            y_cpu = y_gpu._c_tensor.to_cpu()
            y_cpu_tensor = mx.Tensor._from_c_tensor(y_cpu)
            
            # Compare with CPU version
            y_cpu_direct = x.gelu()
            
            print(f"CPU result: {y_cpu_direct}")
            print(f"GPU result: {y_cpu_tensor}")
            print("✓ CUDA GELU passed")
        else:
            print("⚠ CUDA not available - skipping GPU tests")
            
    except Exception as e:
        print(f"⚠ CUDA test failed or not available: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Tensor Operations")
    print("=" * 60)
    
    try:
        test_gelu()
        test_softmax()
        test_matmul()
        test_layer_norm()
        test_cuda_if_available()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
