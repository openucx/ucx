#!/bin/bash
#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

#
# This script makes sure that all options passed can be understood by nvcc.
#

if [ $# -lt 1 ]; then
    echo "Error: NVCC path must be first argument"
    exit 1
fi

nvcc="$1"
shift
output=""
src=""
args=""

# Parse compiler arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -o)
            output="$2"
            shift 2
            ;;
        -c)
            shift
            ;;
        -fPIC)
            args="$args -Xcompiler -fPIC"
            shift
            ;;
        -O*|-I*|-D*|-G|-g|-MD|-MMD|-gencode=*|-std=*)
            args="$args $1"
            shift
            ;;
        -M[FT]|-gencode|-Xcompiler)
            args="$args $1 $2"
            shift 2
            ;;
        -*)
            # Skip other options
            shift
            ;;
        *)
            src="$1"
            shift
            ;;
    esac
done

if [ -z "$output" ];
then
    echo "Error: Missing output file"
    exit 1
fi

if [ -z "$src" ];
then
    echo "Error: Missing source file"
    exit 1
fi

mkdir -p "$(dirname "$output")"
cmd="$nvcc -c $src -o $output $args"
echo $cmd
exec $cmd
