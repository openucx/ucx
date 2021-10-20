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
- Support sharing resources between threads, or allocating dedicated resources
  per thread.
- Event-driven or polling-driven progress.
- Java and Python bindings.
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
- Cray Gemini / Aries (ugni)

Platforms support
*****************
- Supported architectures: x86_64, Arm v8, Power.
- Runs on virtual machines (using SRIOV) and containers (docker, singularity).
- Can utilize either MLNX_OFED or Inbox RDMA drivers.
- Tested on major Linux distributions (RedHat/Ubuntu/SLES).

GPU support
***********
- Cuda (for NVIDIA GPUs)
- ROCm (for AMD GPUs)

Protocols, Optimizations and Advanced Features
**********************************************
- Automatic selection of best transports and devices.
- Zero-copy with registration cache.
- Scalable flow control algorithms.
- Optimized memory pools.
- Accelerated direct-verbs transport for Mellanox devices.
- Pipeline protocols for GPU memory
- QoS and traffic isolation for RDMA transports
- Platform (micro-architecture) specific optimizations (such as memcpy, memory barriers, etc.)
- Multi-rail and RoCE link aggregation group support
- Bare-metal, containers and cloud environments support
- Advanced protocols for transfer messages of different sizes
