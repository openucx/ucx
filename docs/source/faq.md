# Frequently Asked Questions

## General

### Overview  

#### What is UCX?
UCX is a framework (collection of libraries and interfaces) that provides efficient 
and relatively easy way to construct widely used HPC protocols: MPI tag matching, 
RMA operations, rendezvous protocols, stream, fragmentation, remote atomic operations, etc.

#### What is UCP, UCT, UCS?
* **UCT** is a transport layer that abstracts the differences across various hardware architectures and provides a low-level API that enables the implementation of communication protocols. The primary goal of the layer is to provide direct and efficient access to hardware network resources with minimal software overhead. For this purpose, UCT relies on low-level drivers such as uGNI, Verbs, shared memory, ROCM, CUDA. In addition, the layer provides constructs for communication context management (thread-based and application level), and allocation and management of device-specific memories including those found in accelerators. In terms of communication APIs, UCT defines interfaces for immediate (short), buffered copy-and-send (bcopy), and zero-copy (zcopy) communication operations. The short operations are optimized for small messages that can be posted and completed in place. The bcopy operations are optimized for medium size messages that are typically sent through a so-called bouncing-buffer. Finally, the zcopy operations expose zero-copy memory-to-memory communication semantics.

* **UCP** implements higher-level protocols that are typically used by message passing (MPI) and PGAS programming models by using lower-level capabilities exposed through the UCT layer.
UCP is responsible for the following functionality: initialization of the library, selection of transports for communication, message fragmentation, and multi-rail communication. Currently, the API has the following classes of interfaces: Initialization, Remote Memory Access (RMA) communication, Atomic Memory Operations (AMO), Active Message, Tag-Matching, and Collectives. 

* **UCS** is a service layer that provides the necessary functionality for implementing portable and efficient utilities.

#### How can I contribute?
1. Fork
2. Fix bug or implement a new feature
3. Open Pull Request 

#### How do I get in touch with UCX developers?
Please join our mailing list: https://elist.ornl.gov/mailman/listinfo/ucx-group or 
submit issues on github: https://github.com/openucx/ucx/issues

<br/>

### UCX mission

#### What are the key features of UCX?
* **Open source framework supported by vendors**  
The UCX framework is maintained and supported by hardware vendors in addition to the open source community. Every pull-request is tested and multiple hardware platforms supported by vendors community.

* **Performance, performance, performance!** 
The framework architecture, data structures, and components are designed to provide optimized access to the network hardware.

* **High level API for a broad range HPC programming models.**  
UCX provides a high-level and performance-portable network API. The API targets a variety of programming models ranging from high-performance MPI implementation to Apache Spark. UCP API abstracts differences and fills in the gaps across interconnects implemented in the UCT layer. As a result, implementations of programming models and libraries (MPI, OpenSHMEM, Apache Spark, RAPIDS, etc.) is simplified while providing efficient support for multiple interconnects (uGNI, Verbs, TCP, shared memory, ROCM, CUDA, etc.).

* **Support for interaction between multiple transports (or providers) to deliver messages.**  
For example, UCX has the logic (in UCP) to make 'GPUDirect', IB' and share memory work together efficiently to deliver the data where it is needed without the user dealing with this.

* **Cross-transport multi-rail capabilities.** UCX protocol layer can utilize multiple transports,
 event on different types of hardware, to deliver messages faster, without the need for
 any special tuning.

* **Utilizing hardware offloads for optimized performance**, such as RDMA, Hardware tag-matching
 hardware atomic operations, etc.

#### What protocols are supported by UCX?
UCP implements RMA put/get, send/receive with tag matching, Active messages, atomic operations. In near future we plan to add support for commonly used collective operations.

#### Is UCX replacement for GASNET?
No. GASNET exposes high level API for PGAS programming management that provides symmetric memory management capabilities and build in runtime environments. These capabilities are out of scope of UCX project.
Instead, GASNET can leverage UCX framework for fast end efficient implementation of GASNET for the network technologies support by UCX.

#### What is the relation between UCX and network drivers?
UCX framework does not provide drivers, instead it relies on the drivers provided by vendors. Currently we use: OFA VERBs, Cray's UGNI, NVIDIA CUDA.

#### What is the relation between UCX and OFA Verbs or Libfabrics?
UCX is a middleware communication framework that relies on device drivers, e.g. RDMA, CUDA, ROCM. RDMA and OS-bypass network devices typically implement device drivers using the RDMA-core Linux subsystem that is supported by UCX. Support for other network abstractions can be added based on requests and contributions from the community.

#### Is UCX a user-level driver?
UCX is not a user-level driver. Typically, drivers aim to expose fine-grained access to the network architecture-specific features.
UCX abstracts the differences across various drivers and fill-in the gaps using software protocols for some of the architectures that don't provide hardware level support for all the operations.

<br/>

### Dependencies

#### What stuff should I have on my machine to use UCX?

UCX detects the exiting libraries on the build machine and enables/disables support
for various features accordingly. 
If some of the modules UCX was built with are not found during runtime, they will
be silently disabled.

* **Basic shared memory and TCP support** - always enabled
* **Optimized shared memory** - requires knem or xpmem drivers. On modern kernels also CMA (cross-memory-attach) mechanism will be used.
* **RDMA support** - requires rdma-core or libibverbs library.
* **NVIDIA GPU support** - requires Cuda drives
* **AMD GPU support** - requires ROCm drivers 


#### Does UCX depend on an external runtime environment?
UCX does not depend on an external runtime environment.  

`ucx_perftest` (UCX based application/benchmark) can be linked with an external runtime environment that can be used for remote `ucx_perftest` launch, but this an optional configuration which is only used for environments that do not provide direct access to compute nodes. By default this option is disabled. 

<br/>


### Configuration and tuning

#### How can I specify special configuration and tunings for UCX?

UCX takes parameters from specific **environment variables**, which start with the
prefix `UCX_`.  
> **IMPORTANT NOTE:** Setting UCX environment variables to non-default values
may lead to undefined behavior. The environment variables are mostly intended for
advanced users, or for specific tunings or workarounds recommended by the UCX community.

#### Where can I see all UCX environment variables?

* Running `ucx_info -c` prints all environment variables and their default values.
* Running `ucx_info -cf` prints the documentation for all environment variables.

#### UCX configuration file

UCX looks for a configuration file in `{prefix}/etc/ucx/ucx.conf`, where `{prefix}` is the installation prefix configured during compilation.
It allows customization of the various parameters. An environment variable
has precedence over the value defined in `ucx.conf`.
The file can be created using `ucx_info -Cf`.


<br/>

---
<br/>

## Network capabilities

### Selecting networks and transports

#### Which network devices does UCX use?

By default, UCX tries to use all available devices on the machine, and selects 
best ones based on performance characteristics (bandwidth, latency, NUMA locality, etc).
Setting `UCX_NET_DEVICES=<dev1>,<dev2>,...` would restrict UCX to using **only** 
the specified devices.  
For example:
* `UCX_NET_DEVICES=eth2` - Use the Ethernet device eth2 for TCP sockets transport. 
* `UCX_NET_DEVICES=mlx5_2:1` - Use the RDMA device mlx5_2, port 1

Running `ucx_info -d` would show all available devices on the system that UCX can utilize.

#### Which transports does UCX use?

By default, UCX tries to use all available transports, and select best ones 
according to their performance capabilities and scale (passed as estimated number 
of endpoints to *ucp_init()* API).   
For example:
* On machines with Ethernet devices only, shared memory will be used for intra-node
communication and TCP sockets for inter-node communication.
* On machines with RDMA devices, RC transport will be used for small scale, and 
 DC transport (available with Connect-IB devices and above) will be used for large
 scale. If DC is not available, UD will be used for large scale.
* If GPUs are present on the machine, GPU transports will be enabled for detecting
  memory pointer type and copying to/from GPU memory.  

It's possible to restrict the transports in use by setting `UCX_TLS=<tl1>,<tl2>,...`.
The list of all transports supported by UCX on the current machine can be generated
by `ucx_info -d` command.
> **IMPORTANT NOTE**
> In some cases restricting the transports can lead to unexpected and undefined behavior:
> * Using *rc_verbs* or *rc_mlx5* also requires *ud_verbs* or *ud_mlx5* transport for bootstrap.
> * Applications using GPU memory must also specify GPU transports for detecting and
>   handling non-host memory.

In addition to the built-in transports it's possible to use aliases which specify multiple transports.  

##### List of main transports and aliases
<table>
<tr><td>all</td><td>use all the available transports.</td></tr>
<tr><td>sm or shm</td><td>all shared memory transports.</td></tr>
<tr><td>ugni</td><td>ugni_rdma and ugni_udt.</td></tr>
<tr><td>rc</td><td>RC (=reliable connection), "accelerated" transports are used if possible.</td></tr>
<tr><td>ud</td><td>UD (=unreliable datagram), "accelerated" is used if possible.</td></tr>
<tr><td>dc</td><td>DC - Mellanox scalable offloaded dynamic connection transport</td></tr>
<tr><td>rc_x</td><td>Same as "rc", but using accelerated transports only</td></tr>
<tr><td>rc_v</td><td>Same as "rc", but using Verbs-based transports only</td></tr>
<tr><td>ud_x</td><td>Same as "ud", but using accelerated transports only</td></tr>
<tr><td>ud_v</td><td>Same as "ud", but using Verbs-based transports only</td></tr>
<tr><td>cuda</td><td>CUDA (NVIDIA GPU) memory support: cuda_copy, cuda_ipc, gdr_copy</td></tr>
<tr><td>rocm</td><td>ROCm (AMD GPU) memory support: rocm_copy, rocm_ipc, rocm_gdr</td></tr>
<tr><td>tcp</td><td>TCP over SOCK_STREAM sockets</td></tr>
<tr><td>self</td><td>Loopback transport to communicate within the same process</td></tr>
</table>
 
For example:
- `UCX_TLS=rc` will select RC, UD for bootstrap, and prefer accelerated transports
- `UCX_TLS=rc,cuda` will select RC along with Cuda memory transports.


<br/>


### Multi-rail

#### Does UCX support multi-rail?

Yes.

#### What is the default behavior in a multi-rail environment?

By default UCX would pick the 2 best network devices, and split large 
messages between the rails. For example, in a 100MB message - the 1st 50MB
would be sent on the 1st device, and the 2nd 50MB would be sent on the 2nd device.
If the device network speeds are not the same, the split will be proportional to
their speed ratio.

The devices to use are selected according to best network speed, PCI bandwidth, 
and NUMA locality.

#### Is it possible to use more than 2 rails?

Yes, by setting `UCX_MAX_RNDV_RAILS=<num-rails>`. Currently up to 4 are supported.

#### Is it possible that each process would just use the closest device?

Yes, by `UCX_MAX_RNDV_RAILS=1` each process would use a single network device
according to NUMA locality.

#### Can I disable multi-rail?

Yes, by setting `UCX_NET_DEVICES=<dev>` to the single device that should be used.

<br/>

### Adaptive routing

#### Does UCX support adaptive routing fabrics?

Yes.

#### What do I need to do to run UCX with adaptive routing?

When adaptive routing is configured on an Infiniband fabric, it is enabled per SL 
(IB Service Layer).  
Setting `UCX_IB_SL=<sl-num>` will make UCX run on the given
service level and utilize adaptive routing. 

<br/>

### RoCE

#### How to specify service level with UCX?

Setting `UCX_IB_SL=<sl-num>` will make UCX run on the given service level.

#### How to specify DSCP priority?

Setting `UCX_IB_TRAFFIC_CLASS=<num>`.

#### How to specify which address to use?

Setting `UCX_IB_GID_INDEX=<num>` would make UCX use the specified GID index on
the RoCE port. The system command `show_gids` would print all available addresses
and their indexes. 

---
<br/>

## Working with GPU

### GPU support

#### How UCX supports GPU?

UCX protocol operations can work with GPU memory pointers the same way as with Host 
memory pointers. For example, the 'buffer' argument passed to `ucp_tag_send_nb()` can
be either host or GPU memory.  


#### Which GPUs are supported?

Currently UCX supports NVIDIA GPUs by Cuda library, and AMD GPUs by ROCm library.  


#### Which UCX APIs support GPU memory?

Currently only UCX tagged APIs (ucp_tag_send_XX/ucp_tag_recv_XX) and stream APIs 
(ucp_stream_send/ucp_stream_recv_XX) support GPU memory.

#### How to run UCX with GPU support?

In order to run UCX with GPU support, you will need an application which allocates
GPU memory (for example,
[MPI OSU benchmarks with Cuda support](https://mvapich.cse.ohio-state.edu/benchmarks)),
and UCX compiled with GPU support. Then you can run the application as usual (for
example, with MPI) and whenever GPU memory is passed to UCX, it either use GPU-direct
for zero copy operations, or copy the data to/from host memory.
> NOTE When specifying UCX_TLS explicitly, must also specify cuda/rocm for GPU memory
> support, otherwise the GPU memory will not be recognized.
> For example: `UCX_TLS=rc,cuda` or `UCX_TLS=dc,rocm`

#### I'm running UCX with GPU memory and geting a segfault, why?

Most likely UCX does not detect that the pointer is a GPU memory and tries to
access it from CPU. It can happen if UCX is not compiled with GPU support, or fails
to load CUDA or ROCm modules due to missing library paths or version mismatch.
Please run `ucx_info -d | grep cuda` or `ucx_info -d | grep rocm` to check for
UCX GPU support.

#### What are the current limitations of using GPU memory?

* **Static compilation** - programs which are statically compiled with Cuda libraries
  must disable memory detection cache by setting `UCX_MEMTYPE_CACHE=n`. The reason
  is that memory allocation hooks do not work with static compilation. Disabling this
  cache could have a negative effect on performance, especially for small messages.

<br/>

### Performance considerations

#### Does UCX support zero-copy for GPU memory over RDMA?

Yes. For large messages UCX can transfer GPU memory using zero-copy RDMA using
rendezvous protocol. It requires the peer memory q for the relevant GPU type
to be loaded on the system.
> **NOTE:** In some cases if the RDMA network device and the GPU are not on
the same NUMA node, such zero-copy transfer is inefficient.

---
<br/>

## Introspection

### Protocol selection

#### How can I tell which protocols and transports are being used for communication?
  - Set `UCX_LOG_LEVEL=info` to print basic information about transports and devices:
    ```console
     $ mpirun -x UCX_LOG_LEVEL=info -np 2 --map-by node osu_bw D D
     [1645203303.393917] [host1:42:0]     ucp_context.c:1782 UCX  INFO  UCP version is 1.13 (release 0)
     [1645203303.485011] [host2:43:0]     ucp_context.c:1782 UCX  INFO  UCP version is 1.13 (release 0)
     [1645203303.701062] [host1:42:0]          parser.c:1918 UCX  INFO  UCX_* env variable: UCX_LOG_LEVEL=info
     [1645203303.758427] [host2:43:0]          parser.c:1918 UCX  INFO  UCX_* env variable: UCX_LOG_LEVEL=info
     [1645203303.759862] [host2:43:0]      ucp_worker.c:1877 UCX  INFO  ep_cfg[2]: tag(self/memory0 knem/memory cuda_copy/cuda rc_mlx5/mlx5_0:1)
     [1645203303.760167] [host1:42:0]      ucp_worker.c:1877 UCX  INFO  ep_cfg[2]: tag(self/memory0 knem/memory cuda_copy/cuda rc_mlx5/mlx5_0:1)
     # MPI_Init() took 500.788 msec
     # OSU MPI-CUDA Bandwidth Test v5.6.2
     # Send Buffer on DEVICE (D) and Receive Buffer on DEVICE (D)
     # Size    Bandwidth (MB/s)
     [1645203303.805848] [host2:43:0]      ucp_worker.c:1877 UCX  INFO  ep_cfg[3]: tag(rc_mlx5/mlx5_0:1)
     [1645203303.873362] [host1:42:a]      ucp_worker.c:1877 UCX  INFO  ep_cfg[3]: tag(rc_mlx5/mlx5_0:1)
     ...
     ```

  - When using protocols v2, set `UCX_PROTO_INFO=y` for detailed information:
     ```console
     $ mpirun -x UCX_PROTO_ENABLE=y -x UCX_PROTO_INFO=y -np 2 --map-by node osu_bw D D
     [1645027038.617078] [host1:42:0]   +---------------+---------------------------------------------------------------------------------------------------+
     [1645027038.617101] [host1:42:0]   | mpi ep_cfg[2] | tagged message by ucp_tag_send*() from host memory                                                |
     [1645027038.617104] [host1:42:0]   +---------------+--------------------------------------------------+------------------------------------------------+
     [1645027038.617107] [host1:42:0]   |       0..8184 | eager short                                      | self/memory0                                   |
     [1645027038.617110] [host1:42:0]   |    8185..9806 | eager copy-in copy-out                           | self/memory0                                   |
     [1645027038.617112] [host1:42:0]   |     9807..inf | (?) rendezvous zero-copy flushed write to remote | 55% on knem/memory and 45% on rc_mlx5/mlx5_0:1 |
     [1645027038.617115] [host1:42:0]   +---------------+--------------------------------------------------+------------------------------------------------+
     [1645027038.617307] [host2:43:0]   +---------------+---------------------------------------------------------------------------------------------------+
     [1645027038.617337] [host2:43:0]   | mpi ep_cfg[2] | tagged message by ucp_tag_send*() from host memory                                                |
     [1645027038.617341] [host2:43:0]   +---------------+--------------------------------------------------+------------------------------------------------+
     [1645027038.617344] [host2:43:0]   |       0..8184 | eager short                                      | self/memory0                                   |
     [1645027038.617348] [host2:43:0]   |    8185..9806 | eager copy-in copy-out                           | self/memory0                                   |
     [1645027038.617351] [host2:43:0]   |     9807..inf | (?) rendezvous zero-copy flushed write to remote | 55% on knem/memory and 45% on rc_mlx5/mlx5_0:1 |
     [1645027038.617354] [host2:43:0]   +---------------+--------------------------------------------------+------------------------------------------------+
     # MPI_Init() took 1479.255 msec
     # OSU MPI-CUDA Bandwidth Test v5.6.2
     # Size    Bandwidth (MB/s)
     [1645027038.674035] [host2:43:0]   +---------------+--------------------------------------------------------------+
     [1645027038.674043] [host2:43:0]   | mpi ep_cfg[3] | tagged message by ucp_tag_send*() from host memory           |
     [1645027038.674047] [host2:43:0]   +---------------+-------------------------------------------+------------------+
     [1645027038.674049] [host2:43:0]   |       0..2007 | eager short                               | rc_mlx5/mlx5_0:1 |
     [1645027038.674052] [host2:43:0]   |    2008..8246 | eager zero-copy copy-out                  | rc_mlx5/mlx5_0:1 |
     [1645027038.674055] [host2:43:0]   |   8247..17297 | eager zero-copy copy-out                  | rc_mlx5/mlx5_0:1 |
     [1645027038.674058] [host2:43:0]   |    17298..inf | (?) rendezvous zero-copy read from remote | rc_mlx5/mlx5_0:1 |
     [1645027038.674060] [host2:43:0]   +---------------+-------------------------------------------+------------------+
     [1645027038.680982] [host2:43:0]   +---------------+------------------------------------------------------------------------------------+
     [1645027038.680993] [host2:43:0]   | mpi ep_cfg[3] | tagged message by ucp_tag_send*() from cuda/GPU0                                   |
     [1645027038.680996] [host2:43:0]   +---------------+-----------------------------------------------------------------+------------------+
     [1645027038.680999] [host2:43:0]   |       0..8246 | eager zero-copy copy-out                                        | rc_mlx5/mlx5_0:1 |
     [1645027038.681001] [host2:43:0]   |  8247..811555 | eager zero-copy copy-out                                        | rc_mlx5/mlx5_0:1 |
     [1645027038.681004] [host2:43:0]   |   811556..inf | (?) rendezvous pipeline cuda_copy, fenced write to remote, cuda | rc_mlx5/mlx5_0:1 |
     [1645027038.681007] [host2:43:0]   +---------------+-----------------------------------------------------------------+------------------+
     [1645027038.693843] [host1:42:a]   +---------------+--------------------------------------------------------------+
     [1645027038.693856] [host1:42:a]   | mpi ep_cfg[3] | tagged message by ucp_tag_send*() from host memory           |
     [1645027038.693858] [host1:42:a]   +---------------+-------------------------------------------+------------------+
     [1645027038.693861] [host1:42:a]   |       0..2007 | eager short                               | rc_mlx5/mlx5_0:1 |
     [1645027038.693863] [host1:42:a]   |    2008..8246 | eager zero-copy copy-out                  | rc_mlx5/mlx5_0:1 |
     [1645027038.693865] [host1:42:a]   |   8247..17297 | eager zero-copy copy-out                  | rc_mlx5/mlx5_0:1 |
     [1645027038.693867] [host1:42:a]   |    17298..inf | (?) rendezvous zero-copy read from remote | rc_mlx5/mlx5_0:1 |
     [1645027038.693869] [host1:42:a]   +---------------+-------------------------------------------+------------------+
     ...
     ```
