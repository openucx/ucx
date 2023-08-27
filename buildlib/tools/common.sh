#!/bin/bash -eExl

WORKSPACE=${WORKSPACE:=$PWD}
# build in local directory which goes away when docker exits
ucx_build_dir=$HOME/${BUILD_ID}/build
ucx_inst=$ucx_build_dir/install
CUDA_MODULE="dev/cuda11.4"
GDRCOPY_MODULE="dev/gdrcopy2.3_cuda11.4"
JDK_MODULE="dev/jdk"
MVN_MODULE="dev/mvn"
XPMEM_MODULE="dev/xpmem-90a95a4"
PGI_MODULE="hpc-sdk/nvhpc/21.2"
GCC_MODULE="dev/gcc-10.1.0"
ARM_MODULE="arm-compiler/armcc-22.1"
INTEL_MODULE="intel/ics-19.1.1"
FUSE3_MODULE="dev/fuse-3.10.5"

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

MAKE="make V=1"
MAKEP="make V=1 -j${parallel_jobs}"
export AUTOMAKE_JOBS=$parallel_jobs

#
# Prepare build environment
#
prepare() {
	echo " ==== Prepare ===="
	env
	cd ${WORKSPACE}
	if [ -d build-test ]
	then
		chmod u+rwx build-test -R
		rm -rf build-test
	fi
	./autogen.sh
	mkdir -p build-test
	cd build-test
}

#
# Configure and build
#   $1 - mode (devel|release)
#
build() {
	mode=$1
	shift

	config_args="--prefix=$ucx_inst --without-java"
	if [ "X$have_cuda" == "Xyes" ]
	then
		config_args+=" --with-iodemo-cuda"
	fi

	../contrib/configure-${mode} ${config_args} "$@"
	make_clean
	$MAKEP
	$MAKEP install
}

#
# Update global server port
#
step_server_port() {
	# Cycle server_port between (server_port_min)..(server_port_max-1)
	server_port=$((server_port + 1))
	server_port=$((server_port >= server_port_max ? server_port_min : server_port))
}

#
# cleanup ucx
#
make_clean() {
	rm -rf ${ucx_inst}
	$MAKEP ${1:-clean}
}

#
# Prepare build environment
#
prepare_build() {
	echo " ==== Prepare Build ===="
	env
	cd ${WORKSPACE}
	if [ -d ${ucx_build_dir} ]
	then
		chmod u+rwx ${ucx_build_dir} -R
		rm -rf ${ucx_build_dir}
	fi
	./autogen.sh
	mkdir -p ${ucx_build_dir}
	cd ${ucx_build_dir}
	export PROGRESS=0
}
