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
# Check ROCm (amdgpu/kfd) driver is present
#
check_rocm_driver() {
	if [ ! -e "/dev/kfd" ]; then
		azure_log_error "ROCm KFD device /dev/kfd not found"
		exit 1
	fi
}

build() {
	../contrib/configure-devel --enable-gtest --without-valgrind --enable-examples --with-rocm --prefix=$ucx_inst

	make -j$(nproc)
}

test_malloc_hook_mode() {
	mode=$1
	filter=${2:-'rocm_hooks.*'}

	export UCX_MEM_ROCM_HOOK_MODE=${mode}

	# Test hooks in gtest for the selected hook mode. Check the exit status
	# explicitly: a login shell (-l) may reset errexit, so a failing gtest
	# would otherwise not fail the script.
	if ! UCX_MEM_LOG_LEVEL=diag ./test/gtest/gtest --gtest_filter="${filter}"
	then
		azure_log_error "rocm memory hooks test failed in ${mode} mode"
		exit 1
	fi

	unset UCX_MEM_ROCM_HOOK_MODE
}

test_malloc_hook() {
	echo "==== Running rocm malloc hooks test, using ELF relocation table ===="
	test_malloc_hook_mode 'reloc'

	echo "==== Running rocm malloc hooks test, using binary instrumentation ===="
	test_malloc_hook_mode 'bistro'

	echo "==== Running rocm malloc hooks test with far jump, using binary instrumentation ===="
	export UCX_MEM_BISTRO_FORCE_FAR_JUMP=y
	test_malloc_hook_mode 'bistro'
	unset UCX_MEM_BISTRO_FORCE_FAR_JUMP

	echo "==== Running rocm malloc hooks test with hooks disabled ===="
	test_malloc_hook_mode 'none' 'rocm_hooks_disabled.*'
}

prepare
build
check_rocm_driver
test_malloc_hook
