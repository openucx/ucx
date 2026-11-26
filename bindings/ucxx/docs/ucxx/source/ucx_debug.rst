UCX Debugging
=============

InfiniBand
----------

System Configuration
~~~~~~~~~~~~~~~~~~~~


``ibdev2netdev`` -- check to ensure at least one IB controller is configured for IPoIB

::

    user@mlnx:~$ ibdev2netdev
    mlx5_0 port 1 ==> ib0 (Up)
    mlx5_1 port 1 ==> ib1 (Up)
    mlx5_2 port 1 ==> ib2 (Up)
    mlx5_3 port 1 ==> ib3 (Up)

``ucx_info -d`` and ``ucx_info -p -u t`` are helpful commands to display what UCX understands about the underlying hardware.
For example, we can check if UCX has been built correctly with ``RDMA`` and if it is available.

::

    user@pdgx:~$ ucx_info -d | grep -i rdma
    # Memory domain: rdmacm
    #     Component: rdmacm
    # Connection manager: rdmacm


    user@dgx:~$ ucx_info -b | grep -i rdma
    #define HAVE_DECL_RDMA_ESTABLISH  1
    #define HAVE_DECL_RDMA_INIT_QP_ATTR 1
    #define HAVE_RDMACM_QP_LESS       1
    #define UCX_CONFIGURE_FLAGS       "--disable-logging --disable-debug --disable-assertions --disable-params-check --prefix=/gpfs/fs1/user/miniforge/envs/ucx-dev --with-sysroot --enable-cma --enable-mt --with-gnu-ld --with-rdmacm --with-verbs --with-cuda=/gpfs/fs1/SHARE/Utils/CUDA/10.2.89.0_440.33.01"
    #define uct_MODULES               ":cuda:ib:rdmacm:cma"


InfiniBand Performance
~~~~~~~~~~~~~~~~~~~~~~

``ucx_perftest`` should confirm InfiniBand bandwidth to be in the 10+ GB/s range

::

    CUDA_VISIBLE_DEVICES=0 UCX_NET_DEVICES=mlx5_0:1 UCX_TLS=rc,cuda_copy ucx_perftest -t tag_bw -m cuda -s 10000000 -n 10 -p 9999 & \
    CUDA_VISIBLE_DEVICES=1 UCX_NET_DEVICES=mlx5_1:1 UCX_TLS=rc,cuda_copy ucx_perftest `hostname` -t tag_bw -m cuda -s 100000000 -n 10 -p 9999

    +--------------+-----------------------------+---------------------+-----------------------+
    |              |       latency (usec)        |   bandwidth (MB/s)  |  message rate (msg/s) |
    +--------------+---------+---------+---------+----------+----------+-----------+-----------+
    | # iterations | typical | average | overall |  average |  overall |   average |   overall |
    +--------------+---------+---------+---------+----------+----------+-----------+-----------+
    +------------------------------------------------------------------------------------------+
    | API:          protocol layer                                                             |
    | Test:         tag match bandwidth                                                        |
    | Data layout:  (automatic)                                                                |
    | Send memory:  cuda                                                                       |
    | Recv memory:  cuda                                                                       |
    | Message size: 100000000                                                                  |
    +------------------------------------------------------------------------------------------+
                10     0.000  9104.800  9104.800   10474.41   10474.41         110         110


``-c`` option is NUMA dependent and sets the CPU Affinity of process for a particular GPU.  CPU Affinity information can be found in ``nvidia-smi topo -m``
::

    user@mlnx:~$  nvidia-smi topo -m
            GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7    mlx5_0  mlx5_1  mlx5_2  mlx5_3  CPU Affinity
    GPU0     X      NV1     NV1     NV2     NV2     SYS     SYS     SYS     PIX     PHB     SYS     SYS     0-19,40-59
    GPU1    NV1      X      NV2     NV1     SYS     NV2     SYS     SYS     PIX     PHB     SYS     SYS     0-19,40-59
    GPU2    NV1     NV2      X      NV2     SYS     SYS     NV1     SYS     PHB     PIX     SYS     SYS     0-19,40-59
    GPU3    NV2     NV1     NV2      X      SYS     SYS     SYS     NV1     PHB     PIX     SYS     SYS     0-19,40-59
    GPU4    NV2     SYS     SYS     SYS      X      NV1     NV1     NV2     SYS     SYS     PIX     PHB     20-39,60-79
    GPU5    SYS     NV2     SYS     SYS     NV1      X      NV2     NV1     SYS     SYS     PIX     PHB     20-39,60-79
    GPU6    SYS     SYS     NV1     SYS     NV1     NV2      X      NV2     SYS     SYS     PHB     PIX     20-39,60-79
    GPU7    SYS     SYS     SYS     NV1     NV2     NV1     NV2      X      SYS     SYS     PHB     PIX     20-39,60-79
    mlx5_0  PIX     PIX     PHB     PHB     SYS     SYS     SYS     SYS      X      PHB     SYS     SYS
    mlx5_1  PHB     PHB     PIX     PIX     SYS     SYS     SYS     SYS     PHB      X      SYS     SYS
    mlx5_2  SYS     SYS     SYS     SYS     PIX     PIX     PHB     PHB     SYS     SYS      X      PHB
    mlx5_3  SYS     SYS     SYS     SYS     PHB     PHB     PIX     PIX     SYS     SYS     PHB      X

    Legend:

      X    = Self
      SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
      NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
      PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
      PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
      PIX  = Connection traversing at most a single PCIe bridge
      NV#  = Connection traversing a bonded set of # NVLinks

NVLink
------

System Configuration
~~~~~~~~~~~~~~~~~~~~


The NVLink connectivity on the system above (DGX-1) is not homogeneous,
some GPUs are connected by a single NVLink connection (NV1, e.g., GPUs 0 and
1), others with two NVLink connections (NV2, e.g., GPUs 1 and 2), and some not
connected at all via NVLink (SYS, e.g., GPUs 3 and 4)."

NVLink Performance
~~~~~~~~~~~~~~~~~~

``ucx_perftest`` should confirm NVLink bandwidth to be in the 20+ GB/s range

::

    CUDA_VISIBLE_DEVICES=0 UCX_TLS=cuda_ipc,cuda_copy,tcp ucx_perftest -t tag_bw -m cuda -s 10000000 -n 10 -p 9999 -c 0 & \
    CUDA_VISIBLE_DEVICES=1 UCX_TLS=cuda_ipc,cuda_copy,tcp ucx_perftest `hostname` -t tag_bw -m cuda -s 100000000 -n 10 -p 9999 -c 1
    +--------------+-----------------------------+---------------------+-----------------------+
    |              |       latency (usec)        |   bandwidth (MB/s)  |  message rate (msg/s) |
    +--------------+---------+---------+---------+----------+----------+-----------+-----------+
    | # iterations | typical | average | overall |  average |  overall |   average |   overall |
    +--------------+---------+---------+---------+----------+----------+-----------+-----------+
    +------------------------------------------------------------------------------------------+
    | API:          protocol layer                                                             |
    | Test:         tag match bandwidth                                                        |
    | Data layout:  (automatic)                                                                |
    | Send memory:  cuda                                                                       |
    | Recv memory:  cuda                                                                       |
    | Message size: 100000000                                                                  |
    +------------------------------------------------------------------------------------------+
                10     0.000  4163.694  4163.694   22904.52   22904.52         240         240


Experimental Debugging
----------------------

A list of problems we have run into along the way while trying to understand performance issues with UCX/UCXX:

- System-wide settings environment variables. For example, we saw a system with ``UCX_MEM_MMAP_HOOK_MODE`` set to ``none``.  Unsetting this env var resolved problems: https://github.com/rapidsai/ucx-py/issues/616 .  One can quickly check system wide variables with ``env|grep ^UCX_``.


- ``sockcm_iface.c:257 Fatal: sockcm_listener: unable to create handler for new connection``.  This is an
  error we've seen when limits are place on the number of file descriptors and occurs when ``SOCKCM`` is
  used for establishing connections.  User have two choices for resolving this issue: increase the
  ``open files`` limit (check ulimit configuration) or use ``RDMACM`` when establishing a connection
  ``UCX_SOCKADDR_TLS_PRIORITY=rdmacm``.  ``RDMACM`` is only available using InfiniBand devices.
