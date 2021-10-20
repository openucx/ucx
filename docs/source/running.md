<!--
  Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
  See file LICENSE for terms.
-->

# Running UCX

## UCX build and install

#### Getting the source


* Download latest code from github:
```
$ git clone https://github.com/openucx/ucx.git ucx
$ cd ucx
$ ./autogen.sh
```

* Alternatively, download and extract one of UCX pre-configured [releases](download):
```
$ wget https://github.com/openucx/ucx/releases/download/v{RELEASE}/ucx-{RELEASE}.tar.gz
$ tar xzf ucx-{RELEASE}.tar.gz
$ cd ucx-{RELEASE}
```

* (This step is only required for OpenPOWER platforms)
  On Ubuntu platform the config.guess file is a bit outdated and does not have
  support for power. In order to resolve the issue you have to download an updated config.guess.
  From the root of the project:
  ```
  $ wget https://github.com/shamisp/ucx/raw/topic/power8-config/config.guess
  ```

#### Building

1. Configure:
  ```
  $ mkdir build
  $ cd build
  $ ../configure --prefix=<ucx-install-path>
  ```
  > **NOTE**: For best performance configuration, use **../contrib/configure-release**
  > instead of **../configure**.
  > This will strip all debugging and profiling code.


2. Build and install:
  ```
  $ make -j4
  $ make install
  ```

<br/>

---
<br/>

## OpenMPI with UCX

[OpenMPI](https://www.open-mpi.org) supports UCX starting from version 3.0, but
it's recommended to use version 4.0 or higher due to stability and performance
improvements.

### Building

1. Get latest-and-greatest OpenMPI version:
  ```
  $ git clone https://github.com/open-mpi/ompi.git
  $ cd ompi
  $ ./autogen.pl
  ```

2. Configure with UCX:
  ```
  $ mkdir build-ucx
  $ cd build-ucx
  $ ../configure --prefix=<ompi-install-path> --with-ucx=<ucx-install-path>
  ```
> **NOTE**: With OpenMPI 4.0 and above, there could be compilation errors from "btl_uct" component.
> This component is not critical for using UCX; so it could be disabled this way:
> ```
> $ ./configure ... --enable-mca-no-build=btl-uct ...
> ```

3. Build:
  ```bash
  $ make
  $ make install
  ```

<br/>

### Running MPI

Example of the command line (with optional flag to select IB device mlx5_0 port 1):
```
$ mpirun -np 2 -mca pml ucx -x UCX_NET_DEVICES=mlx5_0:1 ./app
```
> **IMPORTANT NOTE**: Recent OpenMPI versions contain a BTL component called 'uct',
> which can cause data corruption when enabled, due to conflict on malloc hooks
> between OPAL and UCM.
> In order to work-around this, use one of the following alternatives:
>
> <u>Alternative 1</u>: Disable btl/uct in OpenMPI build configuration:
> ```
> $ ./configure ... --enable-mca-no-build=btl-uct ...
> ```
>
> <u>Alternative 2</u>: Disable btl/uct at runtime
> ```
> $ mpirun -np 2 -mca pml ucx -mca btl ^uct -x UCX_NET_DEVICES=mlx5_0:1 ./app
> ```

<br/>

### Runtime tunings
By default OpenMPI enables build-in transports (BTLs), which may result in additional
software overheads in the OpenMPI progress function. In order to workaround this issue
you may try to disable certain BTLs.
```
$ mpirun -np 2 -mca pml ucx --mca btl ^vader,tcp,openib,uct -x UCX_NET_DEVICES=mlx5_0:1 ./app
```

<br/>

---
<br/>

## MPICH with UCX
UCX is supported in MPICH 3.3 and higher versions.
UCX is already embedded in the MPICH tarball, so you do not need to separately download UCX.

### Building

1. Download mpich-3.3 or higher from https://www.mpich.org

2. Configure with UCX:
```
$ mkdir build
$ cd build
$ ../configure --prefix=<mpich-install-path> --with-device=ch4:ucx
```

3. Build MPICH:
```
$ make -j4
$ make install
```

<br/>

### Running MPI
Example of the command line (with optional flag to select IB device mlx5_0 port 1):
```
$ mpirun -np 2 -env UCX_NET_DEVICES=mlx5_0:1 ./executable
```

## Running in Docker containers
UCX can run in a container, but requires slight adjustments:

* Some transports may be unsupported, depending on the runtime configuration.
To see the available transports use `ucx_info -d`.

* In order to enable shared memory transport, ptrace capability is required. The recommended shared memory size is 8GB. Add the following to `docker run` commandline: `--cap-add CAP_SYS_PTRACE --shm-size="8g"`.

* To use the shared memory between several containers, add this instead: `--cap-add CAP_SYS_PTRACE --ipc host`.

* To enable RDMA:
  * One of: `rdma-core`, `libibverbs` or `MLNX_OFED` packages has to be installed.

  * The corresponding devices must be allowed: `--device=/dev/infiniband/rdma_cm --device=/dev/infiniband/uverbsX` (replace `X` with the device number).

  * `ulimit -l` should be set to a value large enough to accommodate for transport resources and send/receive buffers. At least 128MB is recommended, a higher value may be needed if the application working set is large.
