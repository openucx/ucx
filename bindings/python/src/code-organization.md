[//]: # 
[//]: # Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
[//]: # See file LICENSE for terms.
[//]: # 
## Overview

The goal of the python bindings is to use UCX API for connection
management and data transfer while attempting to be
pythonesque. Specifically, using UCX's UCP listen-connect and
tag_send/recv functionality borrowed heavily from
[ucp_client_server.c](https://github.com/openucx/ucx/blob/master/test/examples/ucp_client_server.c)
and
[ucp_hello_world.c](https://github.com/openucx/ucx/blob/master/test/examples/ucp_hello_world.c).

To ease python-side usage, the bindings also attempt to provide
minimal functionality similar to those familiar with [Future
objects](https://docs.python.org/3/library/concurrent.futures.html#future-objects). Minimally,
the objects returned from send/recv operations can be called with
`done()` or `result()` to query status or get results of transfer
requests. Furthermore, a subset of transfer functions is compatible
with [asyncio](https://docs.python.org/3/library/asyncio.html) style
of programming. For instance, a coroutine can be used at the
server-side to handle incoming connections concurrently with other
coroutines similar to an example
[here](https://asyncio.readthedocs.io/en/latest/tcp_echo.html).

Lastly, some buffer management utilities are provided for transfer of
contiguous python objects, and for simple experiments.

## Functions
1. UCP API Usage
   + Cython definitions + Python bindings
     - [ucp_py.pyx](./ucp_py.pyx)
   + Cython helper layer
     - [ucp_py_ucp_fxns_wrapper.pyx](./ucp_py_ucp_fxns_wrapper.pyx)
   + C definitions
     - [ucp_py_ucp_fxns.c](./ucp_py_ucp_fxns.c)
     - [ucp_py_ucp_fxns.h](./ucp_py_ucp_fxns.h)
     - [common.h](./common.h)
2. Buffer Management
   + Cython helper layer + Python bindings
     - [ucp_py_buffer_helper.pyx](./ucp_py_buffer_helper.pyx)
   + C definitions
     - [buffer_ops.c](./buffer_ops.c)
     - [buffer_ops.h](./buffer_ops.h)
     - [common.h](./common.h)
3. Build
   + [setup.py](./setup.py)

### UCP API Usage

The basic connection model envisioned for UCP python bindings usage is for a
process to call:
 + listen API if it expects connections
   - `.start_server`
 + connect API targeting listening processes
   - `.get_endpoint(server_ip, server_port)`
 + get bidirectional endpoint handles from connections
   - available as part of listen-accept callback @ server
   - returned from `.get_endpoint(server_ip, server_port)` @ client

The envisioned transfer model is to call:
 + send/recv on endpoints
   - `ep.send`, `ep.recv` which return `CommFuture` objects on
     which `.done` or `.result` calls can be made
   - `ep.send_fast`, `ep.recv_fast` which return `ucp_comm_request`
     objects on which *only* `.done` or `.result` calls can be made
   - `ep.send_msg`, `ep.recv_msg` perform the same action but take
     python objects as arguments and provide ways to return python
     objects
   - the expectation with the above calls is that a send operation
     needs data and length, while these are optional for receive
     operations (when not provided, the receives internally allocate
     memory)
 + optionally make explicit progress on outstanding transfers
   - call `.progress()`

The above calls are exposed through classes/functions in
[ucp_py.pyx](./ucp_py.pyx). The corresponding C backend are written in
[ucp_py_ucp_fxns.c](./ucp_py_ucp_fxns.c) and
[ucp_py_ucp_fxns.h](./ucp_py_ucp_fxns.h)

#### Functions exposed
 + `.init()`
   - Initiate ucp context and create a ucp worker (or progress engine
     context)
   - ucp context detects and sets up communication resources
   - worker helps with connection establishment, drives transfer
     operations, executes callbacks
 + `.start_server(py_func, server_port = -1, is_coroutine = False)`
   - setup a process to listen for connections @ `server_port`
   - pass a python function `py_func` to be called when an incoming
     connection gets accepted
   - `py_func` can be a coroutine but must be indicated
     * TODO: Find out how to automatically detect if a function is a
       coroutine
 + `.stop_server()`
   - stop listening for incoming connections
     * TODO: find if outstandidng operations on existing endpoints are
       completed if listener is destroyed during an ongoing transfer
       or transfers are pending
 + `.fin()`
   - destroy ucp context and worker
 + `.get_endpoint(server_ip, server_port)`
   - client connects to a server which is already listening
     * TODO: provide a `max_timeout` parameter that attempts to
       connect for a maximum of `max_timeout` seconds
 + `.destroy_ep(ucp_ep)`
   - close an endpoint
     * TODO: find if outstandidng operations are completed if endpoint
       is destroyed during an ongoing transfer or transfers are
       pending
 + `.progress()`
   - calls underlying ucp_worker_progress call
   - attempts progress and doesn't ensure progress
 + `.get_obj_from_msg(ucp_msg)`
   - some flavors of receive operation return ucp_msg objects
   - this call returns a python object from the ucp_msg object which
     can be used directly by the python program

#### Classes used
 + CommFuture
   - returned in some form as part of `ep.send*`/`ep.recv*` calls
   - `.done` `.result` to check/get result of transfer
   - result is a `ucp_msg` object which can used to return a python
     object under certain conditions
   - if `CommFuture` is called with await, then request status is
     checked and control is yielded if request isn't complete
 + ServerFuture
   - class used primarily to ensure that multiple connections from
     clients can be accepted
   - not visible to the user of python bindings
 + ucp_py_ep
   - the objects of this class is accessible either from listener
     callback or from calling `.get_endpoint`
   - exposes different flavors send/recv methods
 + ucp_comm_request
   - class used to track outstanding send/recv transfers
 + ucp_msg
   - class used to track messages associated with outstanding
     send/recv transfers

### Buffer Management

A single (cython?) class **buffer_region** exposes methods to
allocate/free host and cuda buffers. This class is defined in
[ucp_py_buffer_helper.pyx](./ucp_py_buffer_helper.pyx). The allocated
buffers is stored using *data_buf* structure, which in turn stores the
pointer to the allocated buffer (TODO: Simplify -- this seems
roundabout). **buffer_region** can be used with contiguous python
objects without needing to allocate memory using `alloc_host` or
`alloc_cuda` methods. To use pointers pointing to the start of the
python object, the `populate_ptr(python-object)` method can be
used. If a **buffer_region** object is associated with a receive
operation, `return_obj()` method can be used to obtain the contiguous
received object.

Internally, `alloc_*` methods translate to `malloc` or `cudaMalloc`
and `free_*` to their counterparts. For experimentation purpose and
for data-validation checks, there are `set_*_buffer` methods that
initialize allocated buffers with a specific
character. `check_*_buffer` methods return the number of mismatches in
the buffers with the character provided. `set_*_buffer` methods have
not be tested when python objects are assoicated with
**buffer_region's** *data_buf* structure.
