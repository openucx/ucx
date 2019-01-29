# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2019, Anaconda, Inc. All rights reserved.
# See file LICENSE for terms.

# This file is a copy of what is available in a Cython demo + some additions


from __future__ import absolute_import, print_function

import os
import sys

from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

UCX_DIR = os.environ.get("UCX_PY_UCX_PATH", "/usr/local/")
CUDA_DIR = os.environ.get("UCX_PY_CUDA_PATH", "/usr/local/cuda")


msg = "The path '{}' does not exist. Set the {} environment variable."

if not os.path.exists(UCX_DIR):
    print(msg.format(UCX_DIR, "UCX_PY_UCX_PATH"), file=sys.stderr)
    sys.exit(1)


if not os.path.exists(CUDA_DIR):
    print(msg.format(CUDA_DIR, "UCX_PY_CUDA_PATH"), file=sys.stderr)
    sys.exit(1)


if not os.path.exists("libucp_py_ucp_fxns.a"):
    assert os.system("gcc -shared -fPIC -c ucp_py_ucp_fxns.c -o ucp_py_ucp_fxns.o") == 0
    assert os.system("gcc -shared -fPIC -c buffer_ops.c -o buffer_ops.o") == 0
    assert os.system("ar rcs libucp_py_ucp_fxns.a ucp_py_ucp_fxns.o buffer_ops.o") == 0


ext_modules = cythonize([
    Extension("ucp_py",
              sources=["ucp_py.pyx"],
              include_dirs=[os.getcwd(), UCX_DIR + '/include', CUDA_DIR + '/include'],
              library_dirs=[os.getcwd(), UCX_DIR + '/lib', CUDA_DIR + '/lib64'],
              runtime_library_dirs=[os.getcwd(), UCX_DIR + '/lib', CUDA_DIR + '/lib64'],
              libraries=['ucp_py_ucp_fxns', 'ucp', 'uct', 'ucm', 'ucs', 'cuda', 'cudart']),
])

setup(
    name='ucx_py',
    ext_modules=ext_modules
)
