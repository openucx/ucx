#!/bin/bash -eExl
realdir=$(realpath $(dirname $0))
source ${realdir}/../../az-helpers.sh

#
# Prepare build environment
#
WORKSPACE=${WORKSPACE:=$PWD}
ucx_inst=${WORKSPACE}/install

prepare() {
	echo " ==== Prepare ===="
	env
	cd ${WORKSPACE}
	mkdir -p build-test
	cd build-test
}

#
# Check nvidia driver is installed
#
check_cuda_driver() {
	if [ ! -f "/proc/driver/nvidia/version" ]; then
		azure_log_error "Cuda driver not found"
		exit 1
	fi
}

build() {
	../contrib/configure-devel --enable-gtest --without-valgrind --enable-examples --with-cuda=/usr/local/cuda --prefix=$ucx_inst

	make -j$(nproc)
}

test_malloc_hook_mode() {
	mode=$1
	cuda_dynamic_exe=./test/apps/test_cuda_hook_dynamic
	cuda_static_exe=./test/apps/test_cuda_hook_static

	export UCX_MEM_CUDA_HOOK_MODE=${mode}

	# Run cuda memory hooks with dynamic link
	${cuda_dynamic_exe}

	# Run cuda memory hooks with static link, if exists. If the static
	# library 'libcudart_static.a' is not present, static test will not
	# be built.
	if [ -x ${cuda_static_exe} ]
	then
		${cuda_static_exe} && status="pass" || status="fail"
		[ ${mode} == "bistro" ] && exp_status="pass" || exp_status="fail"
		if [ ${status} == ${exp_status} ]
		then
			echo "Static link with cuda ${status}, as expected"
		else
			echo "Static link with cuda is expected to ${exp_status}, actual: ${status}"
			exit 1
		fi
	fi

	# Test that driver API hooks work in both reloc and bistro modes,
	# since we call them directly from the test
	${cuda_dynamic_exe} -d
	[ -x ${cuda_static_exe} ] && ${cuda_static_exe} -d

	# Test hooks in gtest
	UCX_MEM_LOG_LEVEL=diag ./test/gtest/gtest --gtest_filter='cuda_hooks.*'

	# Test hooks for device-side variables
	./test/apps/test_cuda_get_symbol_address

	unset UCX_MEM_CUDA_HOOK_MODE
}

test_malloc_hook() {
	echo "==== Running malloc hooks test, using ELF relocation table ===="
	test_malloc_hook_mode 'reloc'

	echo "==== Running malloc hooks test, using binary instrumentation ===="
	test_malloc_hook_mode 'bistro'

	echo "==== Running malloc hooks test with far jump, using binary instrumentation ===="
	export UCX_MEM_BISTRO_FORCE_FAR_JUMP=y
	test_malloc_hook_mode 'bistro'
	unset UCX_MEM_BISTRO_FORCE_FAR_JUMP
}

check_gpu 'gpu'
prepare
build
check_cuda_driver
test_malloc_hook
