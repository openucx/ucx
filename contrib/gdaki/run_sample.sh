#!/bin/bash

# Build DOCA with '0001-Trim-dependencies.patch' applied

# Run on Rock as ROOT
#./doca/build/samples/doca_gpunetio_rdma_verbs_write_bw -g 0000:a3:00 -d mlx5_0 -l 70 -c 1.1.60.1

source ./env.sh
./doca/build/samples/doca_gpunetio_rdma_verbs_write_bw -g 0000:a3:00 -d mlx5_0 -l 70 $*
