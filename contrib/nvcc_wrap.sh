#!/bin/bash

# TODO: replace with UCC nvcc wrapper.
if [ $# -lt 1 ]; then
    echo "Error: NVCC path must be first argument"
    exit 1
fi

nvcc="$1"
shift
output=""
src=""
args=""

# Parse remaining arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -o)
            output="$2"
            shift 2
            ;;
        -c)
            shift
            ;;
        -I*)
            args="$args $1"
            shift
            ;;
        -D*)
            args="$args $1"
            shift
            ;;
        -G)
            args="$args $1"
            shift
            ;;
        -g)
            args="$args $1"
            shift
            ;;
        -M[DP])
            args="$args $1"
            shift
            ;;
        -M[FT])
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

if [ -z "$output" ] || [ -z "$src" ]; then
    echo "Error: Missing output or source file"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p $(dirname "$output")

# Call NVCC
cmd="$nvcc -arch=sm_80 -Xcompiler -fPIC -c $src -o $output $args"
echo $cmd
exec $cmd
