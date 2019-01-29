# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2019, Anaconda, Inc. All rights reserved.
# See file LICENSE for terms.

CC = gcc
UCX_PY_UCX_PATH ?= /usr/local
UCX_PY_CUDA_PATH ?= /usr/local/cuda

CPPFLAGS := -I$(UCX_PY_CUDA_PATH)/include -I$(UCX_PY_UCX_PATH)/include
LDFLAGS := -L$(UCX_PY_CUDA_PATH)/lib64 -lcuda -L$(UCX_PY_UCX_PATH)/lib -lucp -luct -lucm -lucs

pybind/libucp_py_ucp_fxns.a: pybind/ucp_py_ucp_fxns.o pybind/buffer_ops.o
	ar rcs $@ $^

pybind/%.o: pybind/%.c
	$(CC) -shared -fPIC $(CPPFLAGS) -c $^ -o $@ $(LDFLAGS)

install: pybind/libucp_py_ucp_fxns.a
	cd pybind && \
	python3 setup.py build_ext && \
	python3 -m pip install -e .

clean:
	rm pybind/*.o pybind/*.a
	rm -rf pybind/build pybind/ucx_py.egg-info
	rm pybind/ucp_py.c
