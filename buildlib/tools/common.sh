#!/bin/bash -eExl

WORKSPACE=${WORKSPACE:=$PWD}
ucx_inst=${WORKSPACE}/install
CUDA_MODULE="dev/cuda-latest"
GDRCOPY_MODULE="dev/gdrcopy-latest"
JDK_MODULE="dev/jdk"
MVN_MODULE="dev/mvn"
XPMEM_MODULE="dev/xpmem-latest"
PGI_MODULE="pgi/19.7"
GCC_MODULE="dev/gcc-10.1.0"
ARM_MODULE="arm-compiler/armcc-19.0"
INTEL_MODULE="intel/ics-19.1.1"

#
# Parallel build command runs with 4 tasks, or number of cores on the system,
# whichever is lowest
#
num_cpus=$(lscpu -p | grep -v '^#' | wc -l)
[ -z $num_cpus ] && num_cpus=1
parallel_jobs=4
[ $parallel_jobs -gt $num_cpus ] && parallel_jobs=$num_cpus
num_pinned_threads=$(nproc)
[ $parallel_jobs -gt $num_pinned_threads ] && parallel_jobs=$num_pinned_threads

MAKE="make"
MAKEP="make -j${parallel_jobs}"
export AUTOMAKE_JOBS=$parallel_jobs

#
# cleanup ucx
#
make_clean() {
	rm -rf ${ucx_inst}
	$MAKEP ${1:-clean}
}
