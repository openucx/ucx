#!/bin/bash

# Source ./env.sh

set -e
./clone_rdma.sh
./clone_ucx.sh
./clone_doca.sh

# rdma already built

./build_doca.sh
./build_ucx.sh
