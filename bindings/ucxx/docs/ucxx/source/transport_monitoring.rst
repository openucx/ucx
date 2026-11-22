Monitoring Transports
=====================

Below is a list of commonly used tools and commands to monitor InfiniBand and CUDA IPC messages:


Infiniband
----------

Monitor InfiniBand packet counters -- this number should dramatically increase when there's InfiniBand traffic:

::

    watch -n 0.1 'cat /sys/class/infiniband/mlx5_*/ports/1/counters/port_xmit_data'


CUDA IPC/NVLink
---------------

Monitor traffic over all GPUs

::

    nvidia-smi nvlink -gt d


Monitor traffic over all GPUs on counter 0

.. note::
    nvidia-smi nvlink -g is now deprecated

::

    # set counters
    nvidia-smi nvlink -sc 0bz
    watch -d 'nvidia-smi nvlink -g 0'


Stats Monitoring of GPUs
::

    dcgmi dmon -e 449
