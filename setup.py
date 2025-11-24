# setup.py
from setuptools import setup, Extension
import sys
import os
import numpy as np
import qkbind
import subprocess

qkbind_include = os.path.dirname(qkbind.__file__)

# Check if CUDA is available
def cuda_available():
    try:
        subprocess.check_output(['nvcc', '--version'])
        return True
    except:
        return False

# Base sources
sources = [
    'csrc/tensor.c',
    'csrc/tensor_bindings.c',
    'csrc/util.c',
    'csrc/dispatch.c',
    'csrc/cpu/activation_cpu.c',
    'csrc/cpu/norm_cpu.c',
    'csrc/cpu/matmul_cpu.c',
]

extra_compile_args = ['-O3', '-std=c11'] if sys.platform != 'win32' else ['/O2']
extra_link_args = []

# Add CUDA support if available
if cuda_available():
    print("CUDA detected - building with GPU support")
    extra_compile_args.append('-DUSE_CUDA')
    sources.extend([
        'csrc/cuda/activation_cuda.cu',
        'csrc/cuda/norm_cuda.cu',
        'csrc/cuda/matmul_cuda.cu',
    ])
    extra_link_args.extend(['-lcudart', '-L/usr/local/cuda/lib64'])
else:
    print("CUDA not found - building CPU-only version")

ext_modules = [
    Extension(
        'tensor_c',
        sources=sources,
        include_dirs=['csrc', qkbind_include, np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name='qkmx',
    version='0.1.6',
    description='Pure C accelerated tensor operations',
    package_dir={'': 'src'},
    packages=['mx'],
    py_modules=['tensor_c'],
    ext_modules=ext_modules,
    python_requires='>=3.7',
)
