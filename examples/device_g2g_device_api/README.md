## UCX device-side API GPU-to-GPU example (CUDA + MPI)

This example launches a CUDA kernel that uses UCX device-side API to perform a GPU-to-GPU PUT between MPI ranks.

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

# Optional: message length in bytes (default 1MB)
export MSG_LEN=$((1<<20))

mpirun -np 2 ./g2g_ucx_device
```

Notes:
- The program maps a CUDA buffer, exchanges remote addresses and rkeys via MPI, creates a `ucp_device_mem_list`, and launches a kernel that calls `ucp_device_put_single` with `UCT_DEVICE_FLAG_NODELAY` and explicit device-side progress to completion.
- Device selection uses local MPI rank modulo number of GPUs.

### Troubleshooting
- Ensure UCX was configured with CUDA support and the device-side headers installed (e.g., `include/ucp/api/device`).
- If dynamic linker cannot find UCX libs at runtime, either set `LD_LIBRARY_PATH` to include `$UCX_PREFIX/lib` or add `RPATH` via `UCX_PREFIX` in the Makefile.



