# UCX Communication Module for Distributed

This is the UCX communication backend for Dask Distributed, providing high-performance communication capabilities using the UCX (Unified Communication X) framework. It enables efficient GPU-to-GPU communication via NVLink (CUDA IPC), InfiniBand support, and various other high-speed interconnects.

## Installation

This package is typically installed as part of the UCXX project build process. It can also be installed separately via conda-forge:

```bash
mamba install -c conda-forge distributed-ucxx
```

Or via PyPI (requires selection of CUDA version):

```bash
pip install distributed-ucxx-cu13  # For CUDA 13.x
pip install distributed-ucxx-cu12  # For CUDA 12.x
```

## Configuration

This package provides its own configuration system that replaces the UCX configuration previously found in the main Distributed package. Configuration can be provided via:

1. **YAML configuration files**: `distributed-ucxx.yaml`
2. **Environment variables**: Using the `DASK_DISTRIBUTED_UCXX_` prefix
3. **Programmatic configuration**: Using Dask's configuration system

### Configuration Schema

The configuration schema is defined in [`distributed-ucxx-schema.yaml`](distributed_ucxx/distributed-ucxx-schema.yaml) and supports various options:

- UCX transport configuration: `tcp`, `nvlink`, `infiniband`, `cuda-copy`, etc.
- RMM configuration: `rmm.pool-size`
- Advanced options: `multi-buffer`, `environment`

### Example Configuration

New schema:

```yaml
distributed-ucxx:
  tcp: true
  nvlink: true
  infiniband: false
  cuda-copy: true
  create-cuda-context: true
  multi-buffer: false
  environment:
    log-level: "info"
  rmm:
    pool-size: "1GB"
```

Legacy schema (may be removed in the future):

```yaml
distributed:
  comm:
    ucx:
      tcp: true
      nvlink: true
      infiniband: false
      cuda-copy: true
      create-cuda-context: true
      multi-buffer: false
      environment:
        log-level: "info"
      rmm:
        pool-size: "1GB"
```

### Environment Variables

New schema:

```bash
export DASK_DISTRIBUTED_UCXX__TCP=true
export DASK_DISTRIBUTED_UCXX__NVLINK=true
export DASK_DISTRIBUTED_UCXX__RMM__POOL_SIZE=1GB
```

Legacy schema (may be removed in the future):

```bash
export DASK_DISTRIBUTED__COMM__UCX__TCP=true
export DASK_DISTRIBUTED__COMM__UCX__NVLINK=true
export DASK_DISTRIBUTED__RMM__POOL_SIZE=1GB
```

### Python Configuration

New schema:

```python
import dask

dask.config.set({
    "distributed-ucxx.tcp": True,
    "distributed-ucxx.nvlink": True,
    "distributed-ucxx.rmm.pool-size": "1GB"
})
```

Legacy schema (may be removed in the future):

```python
import dask

dask.config.set({
    "distributed.comm.ucx.tcp": True,
    "distributed.comm.ucx.nvlink": True,
    "distributed.rmm.pool-size": "1GB"
})
```

## Usage

The package automatically registers itself as a communication backend for Distributed using the entry point `ucxx`. Once installed, you can use it by specifying the protocol:

```python
from distributed import Client

# Connect using UCXX protocol
client = Client("ucxx://scheduler-address:8786")
```

Or when starting a scheduler/worker:

```bash
dask scheduler --protocol ucxx
dask worker ucxx://scheduler-address:8786
```

## Migration from Distributed

If you're migrating from the legacy UCX configuration in the main Distributed package, update your configuration keys:

- `distributed.comm.ucx.*` is now `distributed-ucxx.*`
- `distributed.rmm.pool-size` is now `distributed-ucxx.rmm.pool-size`

The old configuration schema is still valid for convenience, but may be removed in a future version.

## See Also

- [UCXX Project](https://github.com/rapidsai/ucxx)
- [Dask Distributed Documentation](https://distributed.dask.org/)
- [UCX Project Documentation](https://openucx.readthedocs.io/en/master/index.html)
