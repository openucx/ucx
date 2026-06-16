#
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See file LICENSE for terms.
#

check_elf_soname() {
	lib_path=$1
	soname=$2

	if [ ! -f "$lib_path" ]; then
		azure_log_error "Missing library $lib_path"
		exit 1
	fi

	if ! readelf -d "$lib_path" | grep -q "Library soname: \\[$soname\\]"; then
		azure_log_error "Library $lib_path does not have SONAME $soname"
		exit 1
	fi
}

check_elf_needed() {
	binary_path=$1
	needed=$2

	if [ ! -f "$binary_path" ]; then
		azure_log_error "Missing binary $binary_path"
		exit 1
	fi

	if ! readelf -d "$binary_path" | grep -q "Shared library: \\[$needed\\]"; then
		azure_log_error "Binary $binary_path is not linked to $needed"
		exit 1
	fi
}

check_linker_symlink() {
	link_path=$1
	target_pattern=$2

	if [ ! -L "$link_path" ]; then
		azure_log_error "Missing linker symlink $link_path"
		exit 1
	fi

	if ! readlink "$link_path" | grep -q "$target_pattern"; then
		azure_log_error "Linker symlink $link_path does not point to $target_pattern"
		exit 1
	fi
}

check_uct_module_linkage() {
	module=$1
	suffix=$2
	module_path="${ucx_inst}/lib/ucx/libuct_${module}-${suffix}.so.0.0.0"
	shift 2

	check_elf_soname "$module_path" "libuct_${module}-${suffix}.so.0"
	for needed in "$@"; do
		check_elf_needed "$module_path" "$needed"
	done
}

build_soname_suffix() {
	suffix=ci
	foreign_build_dir=${ucx_build_dir}/foreign
	foreign_inst=${ucx_build_dir}/foreign-install
	soname_suffix_check_hw=${soname_suffix_check_hw:-no}
	common_soname_config_args=(
		--without-java
		--without-go
		--without-rocm
		--without-xpmem
		--without-knem
		--disable-doxygen-doc
	)

	if [ "${soname_suffix_check_hw}" = "yes" ]; then
		echo "==== Enable CUDA and IB for SONAME suffix build ===="
		cuda_local_dir="/usr/local/cuda"
		have_gdrcopy=no

		if ! nvidia-smi -L; then
			azure_log_error "SONAME suffix CUDA/IB check requires a GPU"
			exit 1
		fi

		if [ ! -d /dev/infiniband ]; then
			azure_log_error "SONAME suffix CUDA/IB check requires IB devices"
			exit 1
		fi

		if [ -d "$cuda_local_dir" ] &&
		   find "$cuda_local_dir" -name 'libcudart.so.1[2-9]*' | grep -q .; then
			common_soname_config_args+=(--with-cuda=$cuda_local_dir)
		elif az_module_load $CUDA_MODULE; then
			common_soname_config_args+=(--with-cuda)
		else
			azure_log_error "SONAME suffix CUDA/IB check requires CUDA"
			exit 1
		fi

		if [ -w "/dev/gdrdrv" ] && az_module_load $GDRCOPY_MODULE; then
			have_gdrcopy=yes
			common_soname_config_args+=(--with-gdrcopy)
		else
			common_soname_config_args+=(--without-gdrcopy)
		fi

		common_soname_config_args+=(--with-verbs --with-rdmacm)
	else
		common_soname_config_args+=(
			--without-verbs
			--without-rdmacm
			--without-cuda
		)
	fi

	echo "==== Build foreign UCX without SONAME suffix ===="
	mkdir -p $foreign_build_dir
	pushd $foreign_build_dir
	${WORKSPACE}/contrib/configure-release --prefix=$foreign_inst \
		"${common_soname_config_args[@]}"
	$MAKEP
	$MAKEP install
	popd

	echo "==== Build with SONAME suffix and module deepbind ===="
	${WORKSPACE}/contrib/configure-release --prefix=$ucx_inst \
		--enable-gtest \
		--enable-test-apps \
		--with-soname-suffix=$suffix \
		--enable-module-deepbind \
		"${common_soname_config_args[@]}"
	$MAKEP
	$MAKEP install

	grep "#define UCX_MODULE_FILE_SUFFIX \"-$suffix\"" config.h
	grep "#define UCX_MODULE_DLOPEN_DEEPBIND 1" config.h
	grep " -lucp-${suffix}" "${ucx_inst}/lib/pkgconfig/ucx.pc"
	grep " -lucs-${suffix} -lucm-${suffix}" \
		"${ucx_inst}/lib/pkgconfig/ucx-ucs.pc"
	grep " -luct-${suffix}" "${ucx_inst}/lib/pkgconfig/ucx-uct.pc"
	for lib in ucs ucp uct; do
		grep "lib${lib}-${suffix}.so" \
			"${ucx_inst}/lib/cmake/ucx/ucx-targets.cmake"
	done
	if [ "${soname_suffix_check_hw}" = "yes" ]; then
		grep "#define HAVE_CUDA 1" config.h
		grep "#define HAVE_IB 1" config.h
	fi

	for lib in ucm ucs uct ucp; do
		check_elf_soname \
			"${ucx_inst}/lib/lib${lib}-${suffix}.so.0.0.0" \
			"lib${lib}-${suffix}.so.0"
		check_linker_symlink \
			"${ucx_inst}/lib/lib${lib}.so" \
			"lib${lib}-${suffix}\\.so"
	done

	check_uct_module_linkage cma $suffix \
		"libuct-${suffix}.so.0" \
		"libucs-${suffix}.so.0"
	if [ "${soname_suffix_check_hw}" = "yes" ]; then
		for module in cuda ib rdmacm; do
			check_uct_module_linkage $module $suffix \
				"libuct-${suffix}.so.0" \
				"libucs-${suffix}.so.0"
		done
		if [ "${have_gdrcopy}" = "yes" ]; then
			check_uct_module_linkage cuda_gdrcopy $suffix \
				"libuct_cuda-${suffix}.so.0"
		fi
	fi
	check_elf_soname \
		"${ucx_build_dir}/test/gtest/ucs/test_module/.libs/libtest_module-${suffix}.so.0.0.0" \
		"libtest_module-${suffix}.so.0"
	check_elf_needed \
		"${ucx_inst}/lib/libucp-${suffix}.so.0.0.0" \
		"libuct-${suffix}.so.0"
	check_elf_needed \
		"${ucx_inst}/lib/libucp-${suffix}.so.0.0.0" \
		"libucs-${suffix}.so.0"
	for lib in ucp uct ucs; do
		check_elf_needed \
			"${ucx_inst}/bin/ucx_info" \
			"lib${lib}-${suffix}.so.0"
	done
	check_elf_needed \
		"${ucx_build_dir}/test/apps/.libs/libtest_ucx_isolation_plugin.so" \
		"libucp-${suffix}.so.0"

	LD_LIBRARY_PATH="${ucx_inst}/lib:${foreign_inst}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
		"${ucx_build_dir}/test/apps/test_ucx_dlopen_isolation" \
		"${foreign_inst}/lib/libucp.so.0.0.0" \
		"${ucx_build_dir}/test/apps/.libs/libtest_ucx_isolation_plugin.so" \
		"$suffix" deepbind

	UCX_MODULES= \
	UCX_HANDLE_ERRORS=bt \
	GTEST_FILTER=test_sys.module_file_suffix \
		$MAKE -C test/gtest test
}
