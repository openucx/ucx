#!/bin/bash -eExl

realdir=$(realpath $(dirname $0))
source ${realdir}/common.sh
source ${realdir}/../az-helpers.sh

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
		../configure --prefix=$ucx_inst --with-docs-only
		make_clean
		$MAKE  docs
		make_clean # FIXME distclean does not work with docs-only
	fi
}

#
# Build without verbs
#
build_no_verbs() {
	echo "==== Build without IB verbs ===="
	../contrib/configure-release --prefix=$ucx_inst --without-verbs
	make_clean
	$MAKEP
	make_clean distclean
}

#
# Build without numa support check
#
build_disable_numa() {
	echo "==== Check --disable-numa compilation option ===="
	../contrib/configure-release --prefix=$ucx_inst --disable-numa
	make_clean
	$MAKEP
	make_clean distclean
}

#
# Build a package in release mode
#
build_release_pkg() {
	echo "==== Build release ===="
	../contrib/configure-release
	make_clean
	$MAKEP
	$MAKEP distcheck

	# Show UCX info
	./src/tools/info/ucx_info -s -f -c -v -y -d -b -p -w -e -uart -m 20M

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
		dpkg-buildpackage -us -uc
	else
		echo "==== Build RPM ===="
		../contrib/buildrpm.sh -s -b --nodeps --define "_topdir $PWD"
	fi

	# check that UCX version is present in spec file
	cd ${WORKSPACE}
	# extract version from configure.ac and convert to MAJOR.MINOR.PATCH representation
	version=$(grep -P "define\S+ucx_ver" configure.ac | awk '{print $2}' | sed 's,),,' | xargs echo | tr ' ' '.')
	if ! grep -q "$version" ucx.spec.in; then
		azure_log_error "Current UCX version ($version) is not present in ucx.spec.in changelog"
		exit 1
	fi
	cd -

	make_clean distclean
}

#
# Build with Intel compiler
#
build_icc() {
	if az_module_load $INTEL_MODULE && icc -v
	then
		echo "==== Build with Intel compiler ===="
		../contrib/configure-devel --prefix=$ucx_inst CC=icc CXX=icpc
		make_clean
		$MAKEP
		make_clean distclean
		echo "==== Build with Intel compiler (clang) ===="
		../contrib/configure-devel --prefix=$ucx_inst CC=clang CXX=clang++
		make_clean
		$MAKEP
		make_clean distclean
	else
		azure_log_warning "Not building with Intel compiler"
	fi
	az_module_unload $INTEL_MODULE
}

#
# Build with PGI compiler
#
build_pgi() {
	pgi_test_file=$(mktemp ./XXXXXX).c
	echo "int main() {return 0;}" > ${pgi_test_file}

	if az_module_load $PGI_MODULE && pgcc18 --version && pgcc18 ${pgi_test_file} -o ${pgi_test_file}.out
	then
		echo "==== Build with PGI compiler ===="
		# PGI failed to build valgrind headers, disable it for now
		# TODO: Using non-default PGI compiler - pgcc18 which is going to be default
		#       in next versions.
		#       Switch to default CC compiler after pgcc18 is default for pgi module
		../contrib/configure-devel --prefix=$ucx_inst CC=pgcc18 --without-valgrind
		make_clean
		$MAKEP
		make_clean distclean
	else
		azure_log_warning "Not building with PGI compiler"
	fi

	rm -rf ${pgi_test_file} ${pgi_test_file}.out
	az_module_unload $PGI_MODULE
}

#
# Build debug version
#
build_debug() {
	echo "==== Build with --enable-debug option ===="
	../contrib/configure-devel --prefix=$ucx_inst --enable-debug --enable-examples
	make_clean
	$MAKEP
	make_clean distclean
}

#
# Build prof
#
build_prof() {
	echo "==== Build configure-prof ===="
	../contrib/configure-prof --prefix=$ucx_inst
	make_clean
	$MAKEP
	make_clean distclean
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
	../contrib/configure-devel --prefix=$ucx_inst --with-ugni \
		PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$PWD/../contrib/cray-ugni-mock \
		PKG_CONFIG_TOP_BUILD_DIR=$PWD/..
	make_clean
	$MAKEP

	# make sure UGNI transport is enabled
	grep '#define HAVE_TL_UGNI 1' config.h

	$MAKE  distcheck
	make_clean distclean
}

#
# Build CUDA
#
build_cuda() {
	if az_module_load $CUDA_MODULE
	then
		if az_module_load $GDRCOPY_MODULE
		then
			echo "==== Build with enable cuda, gdr_copy ===="
			../contrib/configure-devel --prefix=$ucx_inst --with-cuda --with-gdrcopy
			make_clean
			$MAKEP
			make_clean distclean

			../contrib/configure-release --prefix=$ucx_inst --with-cuda --with-gdrcopy
			make_clean
			$MAKEP
			make_clean distclean
			az_module_unload $GDRCOPY_MODULE
		fi

		echo "==== Build with enable cuda, w/o gdr_copy ===="
		../contrib/configure-devel --prefix=$ucx_inst --with-cuda --without-gdrcopy
		make_clean
		$MAKEP

		az_module_unload $CUDA_MODULE

		echo "==== Running test_link_map with cuda build but no cuda module ===="
		env UCX_HANDLE_ERRORS=bt ./test/apps/test_link_map

		make_clean distclean
	else
		echo "==== Not building with cuda flags ===="
	fi
}

#
# Build with clang compiler
#
build_clang() {
	if which clang > /dev/null 2>&1
	then
		echo "==== Build with clang compiler ===="
		../contrib/configure-devel --prefix=$ucx_inst CC=clang CXX=clang++
		make_clean
		$MAKEP
		$MAKEP install
		make_clean distclean
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
			../contrib/configure-devel --prefix=$ucx_inst
			make_clean
			$MAKEP
			$MAKEP install
			make_clean distclean
			az_module_unload $GCC_MODULE
		fi
	else
		azure_log_warning "Not building with gcc compiler, glibc version is too old ($ldd_ver)"
	fi
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
		../contrib/configure-devel --prefix=$ucx_inst CC=armclang CXX=armclang++
		make_clean
		$MAKEP
		$MAKEP install
		make_clean distclean
	fi

	rm -rf ${armclang_test_file} ${armclang_test_file}.out
	az_module_unload $ARM_MODULE
}

check_inst_headers() {
	echo "==== Testing installed headers ===="

	../contrib/configure-release --prefix=$PWD/install
	make_clean
	$MAKEP install
	../contrib/check_inst_headers.sh $PWD/install/include
	make_clean distclean
}

check_config_h() {
	srcdir=$PWD/../src

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
# Do a given task only if the current worker is supposed to do it.
#
do_task() {
	$@
	echo "##vso[task.setprogress value=$PROGRESS;]Progress Indicator"
	PROGRESS=$((PROGRESS+5))
}

#
# Prepare build environment
#
prepare() {
	echo " ==== Prepare ===="
	az_init_modules
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
	export PROGRESS=0
}

prepare
do_task build_docs
do_task build_disable_numa
do_task build_no_verbs
do_task build_release_pkg
do_task check_inst_headers
do_task check_config_h
do_task build_icc
do_task build_pgi
do_task build_debug
do_task build_prof
do_task build_ugni
do_task build_clang
do_task build_armclang
do_task build_gcc
do_task build_cuda
