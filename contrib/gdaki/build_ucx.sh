#!/bin/bash

# Must do: source ./env.sh

set -e

cd ucx-fork
./autogen.sh
./contrib/configure-devel --prefix=$(pwd)/rfs \
    --with-verbs=$(pwd)/../rdma-core/build \
    --with-cuda=$CUDA_HOME \
    --with-gdrcopy=/opt/mellanox/gdrcopy

make -j
make install
