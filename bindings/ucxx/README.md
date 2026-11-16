# UCXX

UCXX is an object-oriented C++ interface for UCX, with native support for Python bindings.

## Building

### Environment setup

Before starting it is necessary to have the necessary dependencies installed. The simplest way to get started is to install [Miniforge](https://github.com/conda-forge/miniforge) and then to create and activate an environment with the provided development file, for CUDA 13.x:

```
$ conda env create -n ucxx -f conda/environments/all_cuda-133_arch-$(uname -m).yaml
```

And then activate the newly created environment:

```
$ conda activate ucxx
```

#### Faster conda dependency resolution

The procedure aforementioned should complete without issues, but it may be slower than necessary. One alternative to speed up dependency resolution is to install [mamba](https://mamba.readthedocs.io/en/latest/) before creating the new environment. After installing Miniforge, mamba can be installed with:

```
$ conda install -c conda-forge mamba
```

After that, one can proceed as before, but simply replacing `conda` with `mamba` in the environment creation command:

```
$ mamba env create -n ucxx -f conda/environments/all_cuda-133_arch-$(uname -m).yaml
$ conda activate ucxx
```

### Convenience Script

For convenience, we provide the `./build.sh` script. By default, it will build and install both C++ and Python libraries. For a detailed description on available options please check `./build.sh --help`.

Building C++ and Python libraries manually is also possible, see instructions on building [C++](#c) and [Python](#python).

Additionally, there is a `./build_and_run.sh` script that will call `./build.sh` to build everything as well as running C++ and Python tests and a few benchmarks. Similarly, details on existing options can be queried with `./build_and_run.sh`.

### C++

To build and install the C++ library to `${CONDA_PREFIX}`, with Python support and CCCL CUDA buffer support, as well as building all tests and benchmarks with CUDA/CCCL support, run:

```
mkdir cpp/build
cd cpp/build
cmake .. -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
      -DBUILD_TESTS=ON \
      -DBUILD_BENCHMARKS=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DUCXX_ENABLE_PYTHON=ON \
      -DUCXX_ENABLE_CCCL=ON \
      -DUCXX_BENCHMARKS_ENABLE_CUDA=ON \
      -DUCXX_BENCHMARKS_ENABLE_CCCL=ON
make -j install
```

### Python

```
cd python
python setup.py install
```

## Running benchmarks

### C++

Currently there is one C++ benchmark with comprehensive options. It can be found under `cpp/build/benchmarks/ucxx_perftest` and for a full list of options `-h` argument can be used.

The benchmark is composed of two processes: a server and a client. The server must not specify an IP address or hostname and will bind to all available interfaces, whereas the client must specify the IP address or hostname where the server can be reached.

#### Basic Usage

Below is an example of running a server first, followed by the client connecting to the server on the `localhost` (same as `127.0.0.1`). Both processes specify a list of parameters, which are the message size in bytes (`-s 1000000000`), the number of iterations to perform (`-n 10`) and the progress mode (`-P polling`).

```
$ UCX_TCP_CM_REUSEADDR=y ./benchmarks/ucxx_perftest -s 1000000000 -n 10 -P polling &
$ ./benchmarks/ucxx_perftest -s 1000000000 -n 10 -P polling localhost
```

#### CUDA Memory Support

When built with `UCXX_BENCHMARKS_ENABLE_CUDA=ON`, the benchmark supports multiple CUDA memory types using the `-m` flag:

```
# Server with CUDA device memory
$ UCX_TCP_CM_REUSEADDR=y ./benchmarks/ucxx_perftest -m cuda -s 1048576 -n 10 &

# Client with CUDA device memory
$ ./benchmarks/ucxx_perftest -m cuda -s 1048576 -n 10 127.0.0.1

# Server with CUDA managed memory (unified memory)
$ UCX_TCP_CM_REUSEADDR=y ./benchmarks/ucxx_perftest -m cuda-managed -s 1048576 -n 10 &

# Client with CUDA managed memory
$ ./benchmarks/ucxx_perftest -m cuda-managed -s 1048576 -n 10 127.0.0.1

# Server with CUDA async memory (with streams)
$ UCX_TCP_CM_REUSEADDR=y ./benchmarks/ucxx_perftest -m cuda-async -s 1048576 -n 10 &

# Client with CUDA async memory
$ ./benchmarks/ucxx_perftest -m cuda-async -s 1048576 -n 10 127.0.0.1
```

**Available Memory Types:**
- `host` - Standard host memory allocation (default)
- `cuda` - CUDA device memory allocation
- `cuda-managed` - CUDA unified/managed memory allocation
- `cuda-async` - CUDA device memory with asynchronous operations

**Requirements for CUDA Support:**
- UCXX compiled with `UCXX_BENCHMARKS_ENABLE_CUDA=ON` (if building benchmarks)
- CUDA runtime available
- UCX configured with CUDA transport support
- Compatible CUDA devices on both endpoints

It is recommended to use `UCX_TCP_CM_REUSEADDR=y` when binding to interfaces with TCP support to prevent waiting for the process' `TIME_WAIT` state to complete, which often takes 60 seconds after the server has terminated.

#### CCCL Memory Support

When built with `UCXX_ENABLE_CCCL=ON`, `UCXX_BENCHMARKS_ENABLE_CUDA=ON`, and
`UCXX_BENCHMARKS_ENABLE_CCCL=ON`, additional CCCL-based memory types are available:

```
# Server with CCCL device memory pool
$ UCX_TCP_CM_REUSEADDR=y ./benchmarks/ucxx_perftest -m cccl-device -s 1048576 -n 10 &

# Client with CCCL device memory pool
$ ./benchmarks/ucxx_perftest -m cccl-device -s 1048576 -n 10 127.0.0.1

# Server with CCCL shared memory resource
$ UCX_TCP_CM_REUSEADDR=y ./benchmarks/ucxx_perftest -m cccl-shared -s 1048576 -n 10 &

# Client with CCCL shared memory resource
$ ./benchmarks/ucxx_perftest -m cccl-shared -s 1048576 -n 10 127.0.0.1
```

**Additional CCCL Memory Types:**
- `cccl-device` - CCCL device memory pool
- `cccl-shared` - CCCL shared memory resource
- `cccl-cuda-async` - CCCL CUDA async memory resource
- `cccl-cuda-async-managed` - CCCL CUDA async managed memory resource

**Requirements for CCCL Support:**
- UCXX compiled with `UCXX_ENABLE_CCCL=ON`
- Benchmarks compiled with `UCXX_BENCHMARKS_ENABLE_CUDA=ON`
- Benchmarks compiled with `UCXX_BENCHMARKS_ENABLE_CCCL=ON`
- CCCL library available (fetched automatically via CMake)

### Python

Benchmarks are available for both the Python "core" (synchronous) API and the "high-level" (asynchronous) API.

#### Synchronous

```python
# Thread progress without delayed notification NumPy transfer, 100 iterations
# of single buffer with 100 bytes
python -m ucxx.benchmarks.send_recv \
    --backend ucxx-core \
    --object_type numpy \
    --n-iter 100 \
    --n-bytes 100

# Blocking progress without delayed notification RMM transfer between GPUs 0
# and 3, 100 iterations of 2 buffers (using multi-buffer interface) each with
# 1 MiB
python -m ucxx.benchmarks.send_recv \
    --backend ucxx-core \
    --object_type rmm \
    --server-dev 0 \
    --client-dev 3 \
    --n-iter 100 \
    --n-bytes 100 \
    --progress-mode blocking
```

#### Asynchronous

```python
# NumPy transfer, 100 iterations of 8 buffers (using multi-buffer interface)
# each with 100 bytes
python -m ucxx.benchmarks.send_recv \
    --backend ucxx-async \
    --object_type numpy \
    --n-iter 100 \
    --n-bytes 100 \
    --n-buffers 8

# RMM transfer between GPUs 0 and 3, 100 iterations of 2 buffers (using
# multi-buffer interface) each with 1 MiB
python -m ucxx.benchmarks.send_recv \
    --backend ucxx-async \
    --object_type rmm \
    --server-dev 0 \
    --client-dev 3 \
    --n-iter 100 \
    --n-bytes 1MiB \
    --n-buffers 2

# Polling progress mode without delayed notification NumPy transfer,
# 100 iterations of single buffer with 1 MiB
UCXPY_ENABLE_DELAYED_SUBMISSION=0 \
    python -m ucxx.benchmarks.send_recv \
    --backend ucxx-async \
    --object_type numpy \
    --n-iter 100 \
    --n-bytes 1MiB \
    --progress-mode polling
```

## Logging

Logging is independently available for both C++ and Python APIs. Since the Python interface uses the C++ backend, C++ logging can be enabled when running Python code as well.

### C++

The C++ interface reuses the UCX logger and provides the same log levels and can be enabled via the `UCXX_LOG_LEVEL` environment variable. However, it will not enable UCX logging, one must still set `UCX_LOG_LEVEL` for UCX logging. A few examples are below:

```
# Request trace log level
UCXX_LOG_LEVEL=TRACE_REQ

# Debug log level
UCXX_LOG_LEVEL=DEBUG
```

### Python

The UCXX Python interface uses the `logging` library included in Python. The only used levels currently are `INFO` and `DEBUG`, and can be enabled via the `UCXPY_LOG_LEVEL` environment variable. A few examples are below:

```
# Enable Python info log level
UCXPY_LOG_LEVEL=INFO

# Enable Python debug log level, UCXX request trace log level and UCX data log level
UCXPY_LOG_LEVEL=DEBUG UCXX_LOG_LEVEL=TRACE_REQ UCX_LOG_LEVEL=DATA
```
