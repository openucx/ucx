# Overview

Unified Communication X (UCX) provides an optimized communication
layer for Message Passing ([MPI](https://www.mpi-forum.org/)),
[PGAS](http://www.pgas.org/)/[OpenSHMEM](http://www.openshmem.org/)
libraries and RPC/data-centric applications.

UCX utilizes high-speed networks for inter-node communication, and
shared memory mechanisms for efficient intra-node communication.
  
---

# Quick start

The following will download and build UCX v1.6 [release](download):
```
$ wget https://github.com/openucx/ucx/releases/download/v1.6.1/ucx-1.6.1.tar.gz
$ tar xzf ucx-1.6.1.tar.gz
$ cd ucx-1.6.1
$ ./contrib/configure-release --prefix=$PWD/install
$ make -j8 install
```
  
---

# Documentation

* [Running](running)
* <b>API:</b> [HTML](api/v1.6/html) [PDF](api/v1.6/ucx.pdf)  
* [Examples](https://github.com/openucx/ucx/tree/v1.6.x/test/examples)  
* [Frequently Asked Questions (FAQ)](faq)  
* [Troubleshooting](troubleshoot)

---

# Buzz
   [UCX @ OpenSHMEM workshop](http://www.openucx.org/wp-content/uploads/2015/08/UCX_OpenSHMEM_2015.pdf)
  