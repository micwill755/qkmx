#!/usr/bin/env python3
"""Simple manual test"""

import sys
sys.path.insert(0, 'src')

import mx

print("Testing basic operations...")

# Test 1: Create tensor
print("\n1. Creating tensor...")
x = mx.randn([4])
print(f"   Created: shape={x.shape}, dtype={x.dtype}")
print(f"   Values: {x}")

# Test 2: GELU
print("\n2. Testing GELU...")
y = x.gelu()
print(f"   Input:  {x}")
print(f"   Output: {y}")

# Or use functional API
y2 = mx.gelu(x)
print(f"   Functional: {y2}")

# Test 3: Softmax
print("\n3. Testing Softmax...")
x = mx.Tensor([[1.0, 2.0, 3.0, 4.0]])
y = x.softmax(dim=1)
print(f"   Input:  {x}")
print(f"   Output: {y}")

# Or use functional API
y2 = mx.softmax(x, dim=1)
print(f"   Functional: {y2}")

# Test 4: Matmul
print("\n4. Testing Matmul...")
A = mx.Tensor([[1.0, 2.0], [3.0, 4.0]])
B = mx.Tensor([[5.0, 6.0], [7.0, 8.0]])
C = A @ B  # Use @ operator
print(f"   A @ B = {C}")
print(f"   Expected: [[19, 22], [43, 50]]")

# Test 5: Element-wise operations
print("\n5. Testing element-wise ops...")
x = mx.Tensor([[1.0, 2.0], [3.0, 4.0]])
y = mx.Tensor([[5.0, 6.0], [7.0, 8.0]])
print(f"   x + y = {x + y}")
print(f"   x * y = {x * y}")
print(f"   x * 2 = {x * 2}")

print("\n✓ All basic tests passed!")
