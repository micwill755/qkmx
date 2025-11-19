# setup.py
from setuptools import setup, Extension
import sys

import os
import qkbind

qkbind_include = os.path.dirname(qkbind.__file__)

ext_modules = [
    Extension(
        'tensor_c',
        sources=['csrc/tensor.c', 'csrc/tensor_bindings.c', 'csrc/util.c'], 
        include_dirs=['csrc', qkbind_include],
        extra_compile_args=['-O3', '-std=c11'] if sys.platform != 'win32' else ['/O2'],
    ),
]

setup(
    name='qkmx',
    version='0.1.6',
    description='Pure C accelerated tensor operations',
    package_dir={'': 'src'},
    packages=['mx', 'tensor'],
    py_modules=['tensor_c'],
    ext_modules=ext_modules,
    python_requires='>=3.7',
)
