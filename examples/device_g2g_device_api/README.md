## UCX device-side API GPU-to-GPU example (CUDA + MPI)

This example performs an alltoallv using the UCX device-side API from a CUDA kernel. Each rank sends a variable-length segment to every other rank. The kernel issues device-side PUTs to all peers using a `ucp_device_mem_list` bound to per-peer rkeys. Receiver offsets are computed via exchanged `recvdispls` so each sender writes directly to the correct remote offset.

### Prerequisites
- UCX built with CUDA and device-side API support (headers and libs available)

### Build
```bash
cd examples/device_g2g_device_api
make UCX_PREFIX=/path/to/your/ucx/install
```

### Run
```bash
# Optional: base message length per-destination in bytes (default 1MB)
export MSG_LEN=$((1<<20))

mpirun -np 4 ./g2g_ucx_device
```



