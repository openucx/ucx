# General

## Overview  

#### 1. What is UCX ?
UCX is a framework (collection of libraries and interfaces) that provides efficient 
and relatively easy way to construct widely used HPC protocols: MPI tag matching, 
RMA operations, rendezvous protocols, stream, fragmentation, remote atomic operations, etc.

#### 2. What is UCP, UCT, UCS ?
* **UCT** is a transport layer that abstracts the differences across various hardware architectures and provides a low-level API that enables the implementation of communication protocols. The primary goal of the layer is to provide direct and efficient access to hardware network resources with minimal software overhead. For this purpose UCT relies on low-level drivers provided by vendors such as InfiniBand Verbs, Cray's uGNI, libfabrics, etc. In addition, the layer provides constructs for communication context management (thread-based and ap- plication level), and allocation and management of device- specific memories including those found in accelerators. In terms of communication APIs, UCT defines interfaces for immediate (short), buffered copy-and-send (bcopy), and zero- copy (zcopy) communication operations. The short operations are optimized for small messages that can be posted and completed in place. The bcopy operations are optimized for medium size messages that are typically sent through a so- called bouncing-buffer. Finally, the zcopy operations expose zero-copy memory-to-memory communication semantics.

* **UCP** implements higher-level protocols that are typically used by message passing (MPI) and PGAS programming models by using lower-level capabilities exposed through the UCT layer.
UCP is responsible for the following functionality: initialization of the library, selection of transports for communication, message fragmentation, and multi-rail communication. Currently, the API has the following classes of interfaces: Initialization, Remote Memory Access (RMA) communication, Atomic Memory Operations (AMO), Active Message, Tag-Matching, and Collectives. 

* **UCS** is a service layer that provides the necessary func- tionality for implementing portable and efficient utilities. 

#### 3. How can I contribute ?   
1. Fork
2. Fix bug or implement a new feature
3. Open Pull Request 

#### 4. How do I get in touch with UCX developers ?
Please join our mailing list: https://elist.ornl.gov/mailman/listinfo/ucx-group or 
submit issues on github: http://github.com/openucx/ucx/issues

<br/>

## UCX mission

#### 1. What are the key features of UCX?
* **Open source framework supported by vendors**  
The UCX framework is maintained and supported by hardware vendors in addition to the open source community. Every pull-request is tested and multiple hardware platforms supported by vendors community.

* **Performance, performance, performance!** 
The framework design, data structures, and components are design to provide highly optimized access to the network hardware. 

* **High level API for a broad range HPC programming models.**  
UCX provides a high level API implemented in software 'UCP' to fill in the gaps across interconnects. This allows to use a single set of APIs in a library  to implement multiple interconnects. This reduces the level of complexities when implementing libraries such as Open MPI or OpenSHMEM.  Because of this, UCX performance portable  because a single implementation (in Open MPI or OpenSHMEM) will work efficiently on multiple interconnects. (e.g. uGNI, Verbs, libfabrics, etc). 

* **Support for interaction between multiple transports (or providers) to deliver messages.**  
For example, UCX has the logic (in UCP) to make 'GPUDirect', IB' and share memory work together efficiently to deliver the data where is needed without the user dealing with this. 

* **Cross-transport multi-rail capabilities.** UCX protocol layer can utilize multiple transports,
 event on different types of hardware, to deliver messages faster, without the need for
 any special tuning.

* **Utilizing hardware offloads for optimized performance**, such as RDMA, Hardware tag-matching
  hardware atomic operations, etc. 

#### 2. What protocols are supported by UCX ?
UCP implements RMA put/get, send/receive with tag matching, Active messages, atomic operations. In near future we plan to add support for commonly used collective operations.

#### 3. Is UCX replacement for GASNET ?
No. GASNET exposes high level API for PGAS programming management that provides symmetric memory management capabilities and build in runtime environments. These capabilities are out of scope of UCX project.
Instead, GASNET can leverage UCX framework for fast end efficient implementation of GASNET for the network technologies support by UCX.

#### 4. What is the relation between UCX and network drivers ?
UCX framework does not provide drivers, instead it relies on the drivers provided by vendors. Currently we use: OFA VERBs, Cray's UGNI, NVIDIA CUDA.

#### 5. What is the relation between UCX and OFA Verbs or Libfabrics ?
UCX, is a middleware communication layer that relies on vendors provided user level drivers including OFA Verbs or libfabrics (or any other drivers provided by another communities or vendors) to implement high-level protocols which can be used to close functionality gaps between various vendors drivers including various libfabrics providers: coordination across various drivers, multi-rail capabilities, software based RMA, AMOs, tag-matching for transports and drivers that do not support such capabilities natively.

#### 6. Is UCX a user level driver ?  
No. Typically,  Drivers  aim to expose fine-grain access to the network architecture specific features.
UCX abstracts the differences across various drivers and fill-in the gaps using software protocols for some of the architectures that don't provide hardware level support for all the operations.

<br/>

## Dependencies

#### 1. What stuff should I have on my machine to use UCX ?

UCX detects the exiting libraries on the build machine and enables/disables support
for various features accordingly. 
If some of the modules UCX was built with are not found during runtime, they will
be silently disabled.

* **Basic shared memory and TCP support** - always enabled
* **Optimized shared memory** - requires knem or xpmem drivers. On modern kernels also CMA (cross-memory-attach) mechanism will be used.
* **RDMA support** - requires rdma-core or libibverbs library.
* **NVIDIA GPU support** - requires Cuda drives
* **AMD GPU support** - requires ROCm drivers 


#### 2. Does UCX depend on an external runtime environment ?
UCX does not depend on an external runtime environment.  

`ucx_perftest` (UCX based application/benchmark) can be linked with an external runtime environment that can be used for remote `ucx_perftest` launch, but this an optional configuration which is only used for environments that do not provide direct access to compute nodes. By default this option is disabled. 

<br/>


## Configuration and tuning

#### 1. How can I specify special configuration and tunings for UCX?

UCX takes parameters from specific **environment variables**, which start with the
prefix `UCX_`.  
> **IMPORTANT NOTE:** Changing the values of UCX environment variables to non-default
may lead to undefined behavior. The environment variables are mostly indented for
 dvanced users, or for specific tunings or workarounds recommended by UCX community.

#### 2. Where can I see all UCX environment variables?

* Running `ucx_info -c` prints all environment variables and their default values.
* Running `ucx_info -cf` prints the documentation for all environment variables.


<br/>

---
<br/>

# Network capabilities

## Selecting networks and transports

#### 1. Which network devices does UCX use?

By default, UCX tries to use all available devices on the machine, and selects 
best ones based on performance characteristics (bandwidth, latency, NUMA locality, etc).
Setting `UCX_NET_DEVICES=<dev1>,<dev2>,...` would restrict UCX to using **only** 
the specified devices.  
For example:
* `UCX_NET_DEVICES=eth2` - Use the Ethernet device eth2 for TCP sockets transport. 
* `UCX_NET_DEVICES=mlx5_2:1` - Use the RDMA device mlx5_2, port 1

Running `ucx_info -d` would show all available devices on the system that UCX can utilize.

#### 2. Which transports does UCX use?

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
<tr><td>tcp</td><td>TCP over SOCK_STREAM sockets</td></tr>
<tr><td>self</td><td>Loopback transport to communicate within the same process</td></tr>
</table>
 
For example:
- `UCX_TLS=rc` will select RC, UD for bootstrap, and prefer accelerated transports
- `UCX_TLS=rc,cuda_copy,cuda_ipc` will select RC along with Cuda memory transports.


<br/>


## Multi-rail

#### 1. Does UCX support multi-rail?

Yes.

#### 2. What is the default behavior in a multi-rail environment?

By default UCX would pick the 2 best network devices, and split large 
messages between the rails. For example, in a 100MB message - the 1st 50MB
would be sent on the 1st device, and the 2nd 50MB would be sent on the 2nd device.
If the device network speeds are not the same, the split will be proportional to
their speed ratio.

The devices to use are selected according to best network speed, PCI bandwidth, 
and NUMA locality.

#### 3. Is it possible to use more than 2 rails?

Yes, by setting `UCX_MAX_RNDV_RAILS=<num-rails>`. Currently up to 4 are supported.

#### 4. Is it possible that each process would just use the closest device?

Yes, by `UCX_MAX_RNDV_RAILS=1` each process would use a single network device
according to NUMA locality.

#### 5. Can I disable multi-rail?

Yes, by setting `UCX_NET_DEVICES=<dev>` to the single device that should be used.

<br/>

## Adaptive routing

#### 1. Does UCX support adaptive routing fabrics?

Yes.

#### 2. What do I need to do to run UCX with adaptive routing?

When adaptive routing is configured on an Infiniband fabric, it is enabled per SL 
(IB Service Layer).  
Setting `UCX_IB_SL=<sl-num>` will make UCX run on the given
service level and utilize adaptive routing. 

<br/>

## RoCE

#### 1. How to specify service level with UCX ?

Setting `UCX_IB_SL=<sl-num>` will make UCX run on the given service level.

#### 2. How to specify DSCP priority ?

Setting `UCX_IB_TRAFFIC_CLASS=<num>`.

#### 3. How to specify which address to use?

Setting `UCX_IB_GID_INDEX=<num>` would make UCX use the specified GID index on
the RoCE port. The system command `show_gids` would print all available addresses
and their indexes. 

---
<br/>

# Working with GPU

## GPU support

#### 1. How UCX supports GPU ?

UCX protocol operations can work with GPU memory pointers the same way as with Host 
memory pointers. For example, the 'buffer' argument passed to `ucp_tag_send_nb()` can
be either host or GPU memory.  


#### 2. Which GPUs are supported ?

Currently UCX supports NVIDIA GPUs by Cuda library, and AMD GPUs by ROCm library.  


#### 3. Which UCX APIs support GPU memory?

Currently only UCX tagged APIs (ucp_tag_send_XX/ucp_tag_recv_XX) and stream APIs 
(ucp_stream_send/ucp_stream_recv_XX) support GPU memory.
  
#### 4. What are the current limitations of using GPU memory?

* **Static compilation** - programs which are statically compiled with Cuda libraries
  must disable memory detection cache by setting `UCX_MEMTYPE_CACHE=n`. The reason
  is that memory allocation hooks do not work with static compilation. Disabling this
  cache could have a negative effect on performance, especially for small messages.

<br/>

## Performance considerations

#### 1. Does UCX support zero-copy for GPU memory over RDMA?

Yes. For large messages UCX can transfer GPU memory using zero-copy RDMA using
rendezvous protocol. It requires the peer memory driver for the relevant GPU type
to be loaded on the system.
> **NOTE:** In some cases if the RDMA network device and the GPU are not on
the same NUMA node, such zero-copy transfer is inefficient.



<br/>
