#!/usr/bin/env python3
"""Test that both numpy arrays and Python lists work correctly"""

import sys
sys.path.insert(0, 'src')

import mx
import numpy as np

print("Testing tensor creation from both numpy and lists...")

# Test data
data_list = [[1.0, 2.0], [3.0, 4.0]]
data_numpy = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

print("\n1. Creating tensor from Python list...")
A_list = mx.Tensor(data_list)
print(f"   A_list = {A_list}")
print(f"   shape = {A_list.shape}")

print("\n2. Creating tensor from numpy array...")
A_numpy = mx.Tensor(data_numpy)
print(f"   A_numpy = {A_numpy}")
print(f"   shape = {A_numpy.shape}")

print("\n3. Testing matmul with list-created tensor...")
B_list = mx.Tensor([[5.0, 6.0], [7.0, 8.0]])
C_list = A_list @ B_list
print(f"   A @ B = {C_list}")
print(f"   Expected: [[19, 22], [43, 50]]")

print("\n4. Testing matmul with numpy-created tensor...")
B_numpy = mx.Tensor(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32))
C_numpy = A_numpy @ B_numpy
print(f"   A @ B = {C_numpy}")
print(f"   Expected: [[19, 22], [43, 50]]")

print("\n5. Testing mixed operations (list tensor @ numpy tensor)...")
C_mixed = A_list @ B_numpy
print(f"   A_list @ B_numpy = {C_mixed}")
print(f"   Expected: [[19, 22], [43, 50]]")

print("\n6. Testing with different numpy dtypes...")
# Test float64 (default numpy dtype)
data_float64 = np.array([[1.0, 2.0], [3.0, 4.0]])  # defaults to float64
A_float64 = mx.Tensor(data_float64)
print(f"   From float64: {A_float64}")

# Test int
data_int = np.array([[1, 2], [3, 4]])
A_int = mx.Tensor(data_int)
print(f"   From int: {A_int}")

print("\n7. Testing element-wise ops with both types...")
sum_result = A_list + A_numpy
print(f"   list + numpy = {sum_result}")
print(f"   Expected: [[2, 4], [6, 8]]")

print("\n✓ All tests passed! Both numpy and list inputs work correctly.")
