.. 
.. Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
..
.. See file LICENSE for terms.
..

.. _ucx_features:

*****************
UCX main features
*****************

High-level API features
***********************
- Select either a client/server connection establishment (similar to TCP), or
  connect directly by passing remote address blob. 
- Support sharing resources between threads, or allocating dedicated resources per thread.
- Event-driven or polling-driven progress.
- Seamless handling of GPU memory.

Main APIs
---------
- Stream-oriented send/receive operations.
- Tag-matched send/receive.
- Remote memory access.
- Remote atomic operations.

Fabrics support
***************
- RoCE
- InfiniBand
- TCP sockets
- Shared memory (CMA, knem, xpmem, SysV, mmap)
- Cray Gemini

GPU support
***********
- Cuda (for NVIDIA GPUs)
- ROCm (for AMD GPUs)

Protocols and optimizations
***************************
- Automatic selection of best transports and devices.
- Zero-copy with registration cache.
- Scalable flow control algorithms.
- Optimized memory pools.
- Accelerated direct-verbs transport for Mellanox devices.
- Pipeline protocols for GPU memory
