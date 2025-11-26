import mx

# CPU tensors (default)
a = mx.zeros([2, 3])
print(f"a.is_cuda: {a.is_cuda}")

# GPU tensors - specify at creation
b = mx.zeros([2, 3], device='cuda')
print(f"b.is_cuda: {b.is_cuda}")

# Or move after creation
c = a.cuda()
print(f"c.is_cuda: {c.is_cuda}")

# Matmul on GPU
x = mx.randn([2, 3], device='cuda')
y = mx.randn([3, 4], device='cuda')
z = x @ y
print(f"Result shape: {z.shape}")