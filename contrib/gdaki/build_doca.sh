#!/bin/bash

set -e

source env.sh
mkdir doca/build
cd doca/build

meson setup --prefix=$(pwd)/install \
    -Damalgamation_build=true -Denable_gpu_support=true ..

ninja && ninja install
