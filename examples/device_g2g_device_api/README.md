## UCX device-side API GPU-to-GPU example (CUDA + MPI)

This example performs an alltoallv using the UCX device-side API from a CUDA kernel. Each rank sends a variable-length segment to every other rank. The kernel issues device-side PUTs to all peers using a `ucp_device_mem_list` bound to per-peer rkeys. Receiver offsets are computed via exchanged `recvdispls` so each sender writes directly to the correct remote offset.

### Prerequisites
- CUDA toolkit and a CUDA-capable GPU per rank
- MPI implementation providing `mpicxx`
- UCX built with CUDA and device-side API support (headers and libs available)

### Build
```bash
cd examples/device_g2g_device_api
make UCX_PREFIX=/path/to/your/ucx/install
```

### Run
```bash
export UCX_LOG_LEVEL=warn
export UCX_MEMTYPE_CACHE=n   # optional: avoid memtype cache issues in some setups
export UCX_CUDA_IPC_ENABLE_SAME_PROCESS=y

# Optional: base message length per-destination in bytes (default 1MB)
export MSG_LEN=$((1<<20))

mpirun -np 4 ./g2g_ucx_device
```

Notes:
- Each rank allocates `send_buf` sized to the sum of its `sendcounts[dst]` and `recv_buf` sized to the sum of `recvcounts[src]` determined via `MPI_Alltoall` of counts.
- Ranks exchange worker addresses to create endpoints to all peers, then exchange remote base addresses and rkeys. A device mem list binds the local send registration and per-peer rkeys.
- The CUDA kernel loops over all peers and calls `ucp_device_put_single` for each segment with `UCT_DEVICE_FLAG_NODELAY`.
- A self-copy for rank→rank is done with device-to-device memcpy to match alltoallv semantics.
- Validation copies small samples back to host and checks the source-tag byte.
- Device selection uses the local MPI rank modulo the number of GPUs on the node.

### Troubleshooting
- Ensure UCX was configured with CUDA support and the device-side headers installed (e.g., `include/ucp/api/device`).
- If dynamic linker cannot find UCX libs at runtime, either set `LD_LIBRARY_PATH` to include `$UCX_PREFIX/lib` or add `RPATH` via `UCX_PREFIX` in the Makefile.
- Ensure UCX device-side headers are present under `include/ucp/api/device` in your installation.



