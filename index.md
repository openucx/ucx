## Unified Communication X

Unified Communication X (UCX) provides an optimized communication
layer for Message Passing ([MPI](https://www.mpi-forum.org/)),
[PGAS](http://www.pgas.org/)/[OpenSHMEM](http://www.openshmem.org/)
libraries and RPC/data-centric applications.

UCX utilizes high-speed networks for inter-node communication, and
shared memory mechanisms for efficient intra-node communication.

## Installation

### Download and Install latest release (v1.6.1)

```console
$ wget https://github.com/openucx/ucx/releases/download/v1.6.1/ucx-1.6.1.tar.gz
$ tar xzf ucx-1.6.0.tar.gz
$ cd ucx-1.6.1
$ ./contrib/configure-release --prefix=$PWD/install
$ make -j8 install
```

## Documentation

**v1.6 API doc**   [HTML](api/v1.6/html) [PDF](api/v1.6/ucx.pdf)
   [Examples](https://github.com/openucx/ucx/tree/v1.6.x/test/examples)

## Buzz
   [UCX @ OpenSHMEM workshop](http://www.openucx.org/wp-content/uploads/2015/08/UCX_OpenSHMEM_2015.pdf)

## Developer zone
   [Wiki](https://github.com/openucx/ucx/wiki)
