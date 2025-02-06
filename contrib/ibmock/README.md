## What
IB devices mock implementation. Only EFA devices from same process are supported for now, UD and SRD.

## How

### Build
rdma-core (optional)
```
cd rdma-core
./build.sh
```

UCX
```
cd ucx
./autogen.sh
./contrib/configure-devel --with-verbs=$(pwd)/../rdma-core/build --with-efa
make -j && make install
```

IB mock
```
cd ibmock
make INCLUDES=-I$(pwd)/../rdma-core/build/include
```

### Run
```
export LD_LIBRARY_PATH=$(pwd)/../efa_mock/build:$LD_LIBRARY_PATH
UCX_TLS=srd ucx_perftest -t tag_bw -l
UCX_TLS=ud ucx_perftest -t tag_bw -l
ucx_info -d
```
