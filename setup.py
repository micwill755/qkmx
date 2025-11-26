# setup.py
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import numpy as np
import qkbind
import subprocess

qkbind_include = os.path.dirname(qkbind.__file__)


def find_cuda():
    """Find CUDA installation, preferring driver-compatible version"""
    # Check environment variable first
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home and os.path.exists(cuda_home):
        return cuda_home
    
    # Get driver's max supported CUDA version
    driver_version = get_driver_cuda_version()
    
    # Look for CUDA installations
    cuda_paths = []
    
    # Check versioned installations
    if os.path.exists('/usr/local'):
        for entry in os.listdir('/usr/local'):
            if entry.startswith('cuda-'):
                path = os.path.join('/usr/local', entry)
                if os.path.isdir(path):
                    cuda_paths.append(path)
    
    # Add common paths
    for path in ['/usr/local/cuda', '/usr/cuda', '/opt/cuda']:
        if os.path.exists(path) and path not in cuda_paths:
            cuda_paths.append(path)
    
    if not cuda_paths:
        return None
    
    # If we have driver version info, try to find compatible CUDA
    if driver_version:
        driver_major = float(driver_version.split('.')[0])
        
        # Find best compatible version
        compatible_paths = []
        for path in cuda_paths:
            nvcc = os.path.join(path, 'bin', 'nvcc')
            if not os.path.exists(nvcc):
                continue
            
            try:
                result = subprocess.run(
                    [nvcc, '--version'],
                    capture_output=True,
                    text=True,
                    check=True
                )
                import re
                match = re.search(r'release (\d+\.\d+)', result.stdout)
                if match:
                    cuda_version = match.group(1)
                    cuda_major = float(cuda_version.split('.')[0])
                    
                    # Compatible if CUDA version <= driver supported version
                    if cuda_major <= driver_major:
                        compatible_paths.append((path, cuda_version))
            except:
                continue
        
        # Return newest compatible version
        if compatible_paths:
            compatible_paths.sort(key=lambda x: x[1], reverse=True)
            return compatible_paths[0][0]
    
    # Fallback to first available
    return cuda_paths[0] if cuda_paths else None


def check_cuda_available():
    """Check if CUDA is available"""
    cuda_home = find_cuda()
    if cuda_home is None:
        return False
    
    nvcc_path = os.path.join(cuda_home, 'bin', 'nvcc')
    return os.path.exists(nvcc_path)


def get_cuda_version():
    """Get CUDA toolkit version"""
    cuda_home = find_cuda()
    if not cuda_home:
        return None
    
    try:
        nvcc_path = os.path.join(cuda_home, 'bin', 'nvcc')
        result = subprocess.run(
            [nvcc_path, '--version'],
            capture_output=True,
            text=True,
            check=True
        )
        # Parse version from output like "release 13.0, V13.0.123"
        for line in result.stdout.split('\n'):
            if 'release' in line.lower():
                # Extract version number
                import re
                match = re.search(r'release (\d+\.\d+)', line)
                if match:
                    return match.group(1)
        return None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_driver_cuda_version():
    """Get maximum CUDA version supported by driver"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        driver_version = result.stdout.strip().split('\n')[0].strip()
        
        # Driver version to CUDA version mapping (approximate)
        # See: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
        driver_major = int(driver_version.split('.')[0])
        
        if driver_major >= 570:
            return "13.0"  # Driver 570+ supports CUDA 13.0+
        elif driver_major >= 550:
            return "12.4"
        elif driver_major >= 535:
            return "12.2"
        elif driver_major >= 525:
            return "12.0"
        elif driver_major >= 520:
            return "11.8"
        else:
            return "11.0"
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return None


def get_cuda_arch():
    """Auto-detect CUDA architecture from installed GPUs"""
    # Allow manual override: CUDA_ARCH=sm_80 python setup.py install
    arch = os.environ.get('CUDA_ARCH')
    if arch:
        return arch
    
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        compute_cap = result.stdout.strip().split('\n')[0].strip()
        arch = compute_cap.replace('.', '')
        return f'sm_{arch}'
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        return 'sm_70'  # Fallback to Volta/Turing


class CUDAExtension(build_ext):
    """Custom build extension to compile CUDA files"""
    
    def build_extensions(self):
        # Only customize if CUDA is available
        if check_cuda_available():
            self.compiler.src_extensions.append('.cu')
            
            # Save original _compile method
            original_compile = self.compiler._compile
            
            def _compile_cuda(obj, src, ext, cc_args, extra_postargs, pp_opts):
                if src.endswith('.cu'):
                    self.compiler.set_executable('compiler_so', 'nvcc')
                    if isinstance(extra_postargs, dict):
                        extra_postargs = extra_postargs.get('nvcc', [])
                    else:
                        extra_postargs = ['-c', '--compiler-options', "'-fPIC'"]
                else:
                    self.compiler.set_executable('compiler_so', 'gcc')
                    if isinstance(extra_postargs, dict):
                        extra_postargs = extra_postargs.get('cxx', [])
                
                return original_compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
            
            self.compiler._compile = _compile_cuda
        
        build_ext.build_extensions(self)


# Base CPU sources
cpu_sources = [
    'csrc/tensor.c',
    'csrc/tensor_bindings.c',
    'csrc/util.c',
    'csrc/dispatch.c',
    'csrc/cpu/tensor_cpu.c',
    'csrc/cpu/activation_cpu.c',
    'csrc/cpu/norm_cpu.c',
    'csrc/cpu/matmul_cpu.c',
]

# CUDA sources - using Driver API (pure C, no nvcc needed)
cuda_driver_sources = [
    'csrc/cuda_driver/cuda_init.c',
    'csrc/cuda_driver/tensor_cuda.c',
    'csrc/cuda_driver/activation_driver.c',
    'csrc/cuda_driver/norm_driver.c',
    'csrc/cuda_driver/matmul_driver.c',
]

# CUDA Runtime sources (requires nvcc, has C++ issues)
cuda_runtime_sources = [
    'c++src/cuda/activation_cuda.cu',
    'c++src/cuda/norm_cuda.cu',
    'c++src/cuda/matmul_cuda.cu',
]

# Choose which CUDA implementation to use
# Default to Driver API to avoid C++/extern "C" issues
USE_CUDA_DRIVER_API = os.environ.get('USE_CUDA_DRIVER_API', 'yes').lower() == 'yes'
cuda_sources = cuda_driver_sources if USE_CUDA_DRIVER_API else cuda_runtime_sources

# Check CUDA availability
cuda_available = check_cuda_available()

if cuda_available:
    cuda_home = find_cuda()
    if not cuda_home:
        print("CUDA installation not found. Building CPU-only version.")
        cuda_available = False
    else:
        cuda_toolkit_version = get_cuda_version()
        driver_cuda_version = get_driver_cuda_version()
        cuda_arch = get_cuda_arch()
        
        print(f"Using CUDA from: {cuda_home}")
        print(f"CUDA Toolkit version: {cuda_toolkit_version}")
        print(f"Driver supports CUDA: {driver_cuda_version}")
        print(f"GPU architecture: {cuda_arch}")
        
        # Warn if incompatible but don't fail
        if cuda_toolkit_version and driver_cuda_version:
            toolkit_major = float(cuda_toolkit_version.split('.')[0])
            driver_major = float(driver_cuda_version.split('.')[0])
            
            if toolkit_major > driver_major:
                print(f"WARNING: CUDA toolkit {cuda_toolkit_version} > driver supports ({driver_cuda_version})")
                print("This may cause runtime errors. Automatically selected compatible version if available.")
    
if cuda_available:
    print(f"Building with GPU support")
    sources = cpu_sources + cuda_sources
    
    # Determine compile flags based on CUDA API choice
    if USE_CUDA_DRIVER_API:
        print("Using CUDA Driver API (pure C)")
        # Driver API uses gcc, not nvcc
        if sys.platform != 'win32':
            extra_compile_args = ['-O3', '-std=c11', '-fPIC', '-DUSE_CUDA', '-DUSE_CUDA_DRIVER_API']
        else:
            extra_compile_args = ['/O2', '/DUSE_CUDA', '/DUSE_CUDA_DRIVER_API']
        libraries = ['cuda']  # Driver API uses libcuda, not libcudart
        language = 'c'
        cmdclass = {}
    else:
        print("Using CUDA Runtime API (C++)")
        # Runtime API uses nvcc
        nvcc_flags = [
            '-O3',
            '--use_fast_math',
            f'-arch={cuda_arch}',
            '--compiler-options', '-fPIC',
            '-DUSE_CUDA',
            '-std=c++17',
            '--expt-relaxed-constexpr',
            '--expt-extended-lambda',
        ]
        extra_compile_args = {
            'cxx': ['-O3', '-std=c11', '-fPIC', '-DUSE_CUDA'] if sys.platform != 'win32' else ['/O2', '/DUSE_CUDA'],
            'nvcc': nvcc_flags
        }
        libraries = ['cudart']
        language = 'c++'
        cmdclass = {'build_ext': CUDAExtension}
    
    include_dirs = [
        'csrc',
        'csrc/cpu',
        'csrc/cuda_driver',
        qkbind_include,
        np.get_include(),
        os.path.join(cuda_home, 'include'),
    ]
    
    library_dirs = [
        os.path.join(cuda_home, 'lib64'),
        os.path.join(cuda_home, 'lib'),
    ]
    
    extra_link_args = [f'-l{lib}' for lib in libraries]
else:
    print("CUDA not found. Building CPU-only version.")
    sources = cpu_sources
    
    extra_compile_args = ['-O3', '-std=c11', '-fPIC'] if sys.platform != 'win32' else ['/O2']
    include_dirs = ['csrc', 'csrc/cpu', qkbind_include, np.get_include()]
    library_dirs = []
    libraries = []
    extra_link_args = []
    language = 'c'
    cmdclass = {}

ext_modules = [
    Extension(
        'tensor_c',
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language=language,
    ),
]

setup(
    name='qkmx',
    version='0.1.7',
    description='Pure C accelerated tensor operations with optional CUDA support',
    package_dir={'': 'src'},
    packages=['mx'],
    py_modules=['tensor_c'],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires='>=3.7',
)
