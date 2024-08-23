#!/bin/bash -eExl
#
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See file LICENSE for terms.
#

realdir=$(realpath $(dirname $0))
source ${realdir}/common.sh
source ${realdir}/../az-helpers.sh
long_test=${long_test:-no}

#
# Build documentation
#
build_docs() {
	if [ `cat /etc/system-release | grep -i "fedora release 34" | wc -l` -gt 0 ]; then
		azure_log_warning "Skip build docs on Fedora 34"
		return 0
	fi
	doxy_ready=0
	doxy_target_version="1.8.11"
	doxy_version="$(doxygen --version)" || true

	# Try load newer doxygen if native is older than 1.8.11
	if ! (echo $doxy_target_version; echo $doxy_version) | sort -CV
	then
		if az_module_load tools/doxygen-1.8.11
		then
			doxy_ready=1
		fi
	else
		doxy_ready=1
	fi

	if [ $doxy_ready -eq 1 ]
	then
		echo " ==== Build docs only ===="
		${WORKSPACE}/contrib/configure-release --prefix=$ucx_inst --with-docs-only
		$MAKE docs
	fi
}

#
# Build without verbs
#
build_no_verbs() {
	echo "==== Build without IB verbs ===="
	${WORKSPACE}/contrib/configure-release --prefix=$ucx_inst --without-verbs
	$MAKEP
}

#
# Build a package in release mode
#
build_release_pkg() {
	echo "==== Build release ===="
	${WORKSPACE}/contrib/configure-release
	$MAKEP distcheck

	if [ -f /etc/redhat-release -o -f /etc/fedora-release ]; then
		rpm_based=yes
	elif [ `cat /etc/os-release | grep -i "ubuntu\|mint"|wc -l` -gt 0 ]; then
		rpm_based=no
	else
		# try rpm tool to detect distro
		set +e
		out=$(rpm -q rpm 2>/dev/null)
		rc=$?
		set -e
		rpm_based=yes
		if [[ $rc != 0 || "$out" == *"not installed"* ]]; then
			rpm_based=no
		fi
	fi

	if [[ "$rpm_based" == "no" && -x /usr/bin/dpkg-buildpackage ]]; then
		echo "==== Build debian package ===="
		(
			tarball=$(ls -t ucx-*.tar.gz | head -n1)
			tar xzf $tarball
			subdir=${tarball%.tar.gz}
			cd $subdir
			dpkg-buildpackage -us -uc
		)
	else
		echo "==== Build RPM ===="
		echo "$PWD"
		${WORKSPACE}/contrib/buildrpm.sh -s -b --nodeps --define "_topdir $PWD"
		if rpm -qp ${PWD}/rpm-dist/ucx-[0-9]*.rpm --requires | grep cuda; then
			azure_log_error "Release build depends on CUDA while it should not"
			exit 1
		fi
		echo "==== Build debug RPM ===="
		${WORKSPACE}/contrib/buildrpm.sh -s -b -d --nodeps --define "_topdir $PWD/debug"
	fi

	# check that UCX version is present in spec file
	cd ${WORKSPACE}
	# extract version from configure.ac and convert to (MAJOR).(MINOR).(PATCH)(EXTRA) representation
	major_ver=$(grep -P "define\S+ucx_ver_major" configure.ac | awk '{print $2}' | sed 's/)//')
	minor_ver=$(grep -P "define\S+ucx_ver_minor" configure.ac | awk '{print $2}' | sed 's/)//')
	patch_ver=$(grep -P "define\S+ucx_ver_patch" configure.ac | awk '{print $2}' | sed 's/)//')
	extra_ver=$(grep -P "define\S+ucx_ver_extra" configure.ac | awk '{print $2}' | sed 's/)//')
	version=${major_ver}.${minor_ver}.${patch_ver}${extra_ver}
	if ! grep -q "$version" ucx.spec.in; then
		azure_log_error "Current UCX version ($version) is not present in ucx.spec.in changelog"
		exit 1
	fi
	cd -
}

build_icc_check() {
	cc=$1
	cxx=$2

	if $cc -v
	then
		echo "==== Build with Intel compiler $cc ===="
		${WORKSPACE}/contrib/configure-devel --prefix=$ucx_inst CC=$cc CXX=$cxx
		$MAKEP
		make_clean distclean
	else
		azure_log_warning "Not building with Intel compiler $cc"
	fi
}

#
# Build with Intel compiler
#
build_icc() {
	if az_module_load $INTEL_MODULE
	then
		build_icc_check icc icpc
		build_icc_check icx icpx
		build_icc_check clang clang++
	else
		azure_log_warning "Not building with Intel compilers"
	fi
	az_module_unload $INTEL_MODULE
}

#
# Build with PGI compiler
#
build_pgi() {
	if az_module_load $PGI_MODULE
	then
		# add_network_host utility from $PGI_MODULE it create config file for machine
		# Doc: https://docs.nvidia.com/hpc-sdk/hpc-sdk-install-guide/index.html
		add_network_host
		echo "==== Build with PGI compiler ===="
		# PGI failed to build valgrind headers, disable it for now
		# TODO: Using non-default PGI compiler - pgcc18 which is going to be default
		#       in next versions.
		#       Switch to default CC compiler after pgcc18 is default for pgi module
		${WORKSPACE}/contrib/configure-devel --prefix=$ucx_inst --without-valgrind
		$MAKEP
		# TODO: Check why "make distclean" is needed to cleanup after PGI compiler
		make_clean distclean
	else
		azure_log_warning "Not building with PGI compiler"
	fi
	az_module_unload $PGI_MODULE
}

#
# Build debug version
#
build_debug() {
	echo "==== Build with --enable-debug option ===="
	${WORKSPACE}/contrib/configure-devel --prefix=$ucx_inst --enable-debug --enable-examples
	$MAKEP

	# Show UCX info
	./src/tools/info/ucx_info -s -f -c -v -y -d -b -p -w -e -uart
}

#
# Build prof
#
build_prof() {
	echo "==== Build configure-prof ===="
	${WORKSPACE}/contrib/configure-prof --prefix=$ucx_inst
	$MAKEP
}

#
# Build UGNI
#
build_ugni() {
	echo "==== Build with cray-ugni ===="
	#
	# Point pkg-config to contrib/cray-ugni-mock, and replace
	# PKG_CONFIG_TOP_BUILD_DIR with source dir, since the mock .pc files contain
	# relative paths.
	#
	${WORKSPACE}/contrib/configure-devel --prefix=$ucx_inst --with-ugni \
		PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/${WORKSPACE}/contrib/cray-ugni-mock \
		PKG_CONFIG_TOP_BUILD_DIR=${WORKSPACE}
	$MAKEP

	# make sure UGNI transport is enabled
	grep '#define HAVE_TL_UGNI 1' config.h

	$MAKEP distcheck
}

#
# Build CUDA
#
build_cuda() {
	if [[ $CONTAINER == *"rocm"* ]]; then
		echo "==== Not building with cuda flags ===="
		return
	fi

	if az_module_load $CUDA_MODULE
	then
		if az_module_load $GDRCOPY_MODULE
		then
			echo "==== Build with enable cuda, gdr_copy ===="
			${WORKSPACE}/contrib/configure-release --prefix=$ucx_inst --with-cuda --with-gdrcopy
			$MAKEP
			make_clean distclean

			echo "==== Build with enable cuda, gdr_copy by ./configure parameter ===="
			# Use path to CUDA instead of loading CUDA as module, to check
			# GDRCopy subcomponent does not include CUDA headers
			cuda_path=$(dirname $(dirname $(which nvcc)))
			az_module_unload $CUDA_MODULE
			${WORKSPACE}/contrib/configure-devel --prefix=$ucx_inst --with-cuda=$cuda_path --with-gdrcopy
			$MAKEP
			make_clean distclean
			az_module_load $CUDA_MODULE

			az_module_unload $GDRCOPY_MODULE
		fi

		echo "==== Build with enable cuda, w/o gdr_copy ===="
		${WORKSPACE}/contrib/configure-devel --prefix=$ucx_inst --with-cuda --without-gdrcopy
		$MAKEP

		az_module_unload $CUDA_MODULE

		echo "==== Running test_link_map with cuda build but no cuda module ===="
		env UCX_HANDLE_ERRORS=bt ./test/apps/test_link_map
	else
		echo "==== Not building with cuda flags ===="
	fi
}

#
# Build ROCm
#
build_rocm() {
	if [ -f /opt/rocm/bin/rocminfo ]; then
		echo "==== Build with enable rocm  ===="
		${WORKSPACE}/contrib/configure-devel --prefix=$ucx_inst --with-rocm
		$MAKEP
	else
		echo "==== Not building with rocm ===="
	fi
}

#
# Build with clang compiler
#
build_clang() {
	if which clang > /dev/null 2>&1
	then
		echo "==== Build with clang compiler ===="
		${WORKSPACE}/contrib/configure-devel --prefix=$ucx_inst CC=clang CXX=clang++
		$MAKEP
		$MAKEP install
	else
		echo "==== Not building with clang compiler ===="
	fi
}

#
# Build with gcc-latest module
#
build_gcc() {
	#If the glibc version on the host is older than 2.14, don't run
	#check the glibc version with the ldd version since it comes with glibc
	#see https://www.linuxquestions.org/questions/linux-software-2/how-to-check-glibc-version-263103/
	#see https://benohead.com/linux-check-glibc-version/
	#see https://stackoverflow.com/questions/9705660/check-glibc-version-for-a-particular-gcc-compiler
	if [ `cat /etc/os-release | grep -i "ubuntu\|mint"|wc -l` -gt 0 ]; then
		azure_log_warning "Not building with latest gcc compiler on Ubuntu"
		return 0
	fi

	ldd_ver="$(ldd --version | awk '/ldd/{print $NF}')"
	if (echo "2.14"; echo $ldd_ver) | sort -CV
	then
		if az_module_load $GCC_MODULE
		then
			echo "==== Build with GCC compiler ($(gcc --version|head -1)) ===="
			${WORKSPACE}/contrib/configure-devel $@ --prefix=$ucx_inst
			$MAKEP
			$MAKEP install
			az_module_unload $GCC_MODULE
		fi
	else
		azure_log_warning "Not building with gcc compiler, glibc version is too old ($ldd_ver)"
	fi
}

build_no_devx() {
	build_gcc --with-devx=no
}

build_no_openmp() {
	build_gcc --disable-openmp
}

build_gcc_debug_opt() {
	build_gcc CFLAGS=-Og CXXFLAGS=-Og
}

#
# Build with armclang compiler
#
build_armclang() {
	arch=$(uname -m)
	if [ "${arch}" != "aarch64" ]
	then
		echo "==== Not building with armclang compiler on ${arch} ===="
		return 0
	fi

	armclang_test_file=$(mktemp ./XXXXXX).c
	echo "int main() {return 0;}" > ${armclang_test_file}
	if az_module_load $ARM_MODULE && armclang --version && armclang ${armclang_test_file} -o ${armclang_test_file}.out
	then
		echo "==== Build with armclang compiler ===="
		${WORKSPACE}/contrib/configure-devel --prefix=$ucx_inst CC=armclang CXX=armclang++
		$MAKEP
		$MAKEP install
	fi

	rm -rf ${armclang_test_file} ${armclang_test_file}.out
	az_module_unload $ARM_MODULE
}

check_inst_headers() {
	echo "==== Testing installed headers ===="

	${WORKSPACE}/contrib/configure-release --prefix=${ucx_inst}
	$MAKEP install
	${WORKSPACE}/contrib/check_inst_headers.sh ${ucx_inst}/include
}

check_config_h() {
	srcdir=${WORKSPACE}/src

	# Check if all .c files include config.h
	echo "==== Checking for config.h files in directory $srcdir ===="

	missing=`find $srcdir \( -name "*.c" -o -name "*.cc" \) -type f -exec grep -LP '\#\s*include\s+"config.h"' {} \;`

	if [ `echo $missing | wc -w` -eq 0 ]
	then
		echo "Check successful "
	else
		azure_log_error "Missing include config.h in files: $missing"
		exit 1
	fi
}

#
# Test if cmake can correctly find and link ucx
#
build_cmake_examples() {
	echo "==== Build CMake sample ===="

	if which cmake
	then
		${WORKSPACE}/contrib/configure-release --prefix=$ucx_inst
		$MAKEP
		$MAKEP install

		mkdir -p /tmp/cmake-ucx
		pushd /tmp/cmake-ucx
		cmake ${WORKSPACE}/examples/cmake -DCMAKE_PREFIX_PATH=$ucx_inst
		cmake --build .

		if ./test_ucp && ./test_uct
		then
			echo "Check successful "
		else
			azure_log_error "CMake test failed."
			exit 1
		fi
		popd
	else
		azure_log_warning "cmake executable not found, skipping cmake test"
	fi
}

#
# Build with FUSE
#
build_fuse() {
	if az_module_load $FUSE3_MODULE
	then
		echo "==== Build with FUSE (dynamic link) ===="
		${WORKSPACE}/contrib/configure-devel --prefix=$ucx_inst --with-fuse3
		$MAKEP
		make_clean distclean

		echo "==== Build with FUSE (static link) ===="
		${WORKSPACE}/contrib/configure-devel --prefix=$ucx_inst --with-fuse3-static
		$MAKEP

		az_module_unload $FUSE3_MODULE
	else
		azure_log_warning "cannot load FUSE module, skipping build with FUSE"
	fi
}

az_init_modules
prepare_build

tests=('build_docs' \
		'build_debug' \
		'build_prof' \
		'build_ugni' \
		'build_cuda' \
		'build_rocm' \
		'build_no_verbs' \
		'build_release_pkg' \
		'build_cmake_examples' \
		'build_fuse')
if [ "${long_test}" = "yes" ]
then
	tests+=('check_config_h' \
			'check_inst_headers' \
			'build_icc'\
			'build_pgi' \
			'build_gcc' \
			'build_no_devx' \
			'build_no_openmp' \
			'build_gcc_debug_opt' \
			'build_clang' \
			'build_armclang')
fi

num_tests=${#tests[@]}
for ((i=0;i<${num_tests};++i))
do
	test_name=${tests[$i]}

	# update progress indicator
	progress=$(( (i * 100) / num_tests ))
	echo "##vso[task.setprogress value=${progress};]${test_name}"

	# cleanup build dir before the task
	[ -d "${ucx_build_dir}" ] && rm -rf ${ucx_build_dir}/*

	# run the test
	$test_name || { azure_log_error "Test failed: $test_name"; exit 1; }
done
