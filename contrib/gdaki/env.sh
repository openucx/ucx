#!/bin/bash

module load dev/cuda12.9.0
module load dev/gdrcopy2.5_cuda12.9.0
#module load hpcx-gcc

sudo yum install -y glib2-devel libzip-devel json-c-devel \
    libpcap-devel jsoncpp-devel ninja-build
sudo yum install -y dpdk-devel

if [ $(id -u) -ne 0 ]; then
    sudo pip3 install meson>=0.61.2
fi

gdr=$(echo "$INCLUDE" | cut -d: -f 1)
gdr_lib=$(echo "$LIBRARY_PATH" | cut -d: -f1)
ucx=$(pwd)/ucx/rfs
rdma=$(pwd)/rdma-core/rfs
doca=$(pwd)/doca/build/install
sudo ln -sf "$(dirname $gdr)" /opt/mellanox/gdrcopy

export CFLAGS="-I$ucx/include -I$rdma/include -I$gdr -I$doca/include"
export CPPFLAGS="-I$ucx/include -I$rdma/include -I$gdr -I$doca/include"
export CXXFLAGS="-I$ucx/include -I$rdma/include -I$gdr -I$doca/include"
export LDFLAGS="-L$ucx/lib -L$rdma/lib -L$gdr_lib -L$doca/lib64"
export NVCC_FLAGS="-I$doca/include -I$rdma/include"
export LD_LIBRARY_PATH="$rdma/lib:$doca/lib64:$LD_LIBRARY_PATH"
