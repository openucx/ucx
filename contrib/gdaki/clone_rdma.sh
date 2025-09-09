#!/bin/bash

set -eu

[ -d rdma-core ] && { echo "rdma-core already exists"; exit 1; }
git clone https://github.com/linux-rdma/rdma-core.git
cd rdma-core
git checkout v57.0

mkdir build
cd build
cmake -DNO_MAN_PAGES=1 -DCMAKE_INSTALL_PREFIX=$(pwd)/../rfs ../
make -j
make install
cd ../..
