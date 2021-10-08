#!/bin/bash -eExl

realdir=$(realpath $(dirname $0))
source ${realdir}/common.sh
source ${realdir}/../az-helpers.sh

COV_MODULE="tools/cov-2019.12"

#
# Run Coverity and report errors
# The argument is a UCX build type: devel or release
#
modules_for_coverity() {
	res=0
	az_module_load $COV_MODULE
	res=$(($res+$?))
	az_module_load $CUDA_MODULE
	res=$(($res+$?))
	az_module_load $GDRCOPY_MODULE
	res=$(($res+$?))
	az_module_load $JDK_MODULE
	res=$(($res+$?))
	az_module_load $MVN_MODULE
	res=$(($res+$?))
	az_module_load $XPMEM_MODULE
	res=$(($res+$?))
	return $res
}

modules_for_coverity_unload() {
	res=0
	az_module_unload $COV_MODULE
	res=$(($res+$?))
	az_module_unload $CUDA_MODULE
	res=$(($res+$?))
	az_module_unload $GDRCOPY_MODULE
	res=$(($res+$?))
	az_module_unload $JDK_MODULE
	res=$(($res+$?))
	az_module_unload $MVN_MODULE
	res=$(($res+$?))
	az_module_unload $XPMEM_MODULE
	res=$(($res+$?))
	return $res
}

run_coverity() {

	az_init_modules
	modules_for_coverity

	ucx_build_type=$1

	xpmem_root=$(module show $XPMEM_MODULE 2>&1 | awk '/CPATH/ {print $3}' | sed -e 's,/include,,')
	with_xpmem="--with-xpmem=$xpmem_root"

	${WORKSPACE}/contrib/configure-$ucx_build_type --prefix=$ucx_inst --with-cuda --with-gdrcopy --with-java $with_xpmem
	cov_build_id="cov_build_${ucx_build_type}"
	cov_build="$ucx_build_dir/$cov_build_id"
	rm -rf $cov_build
	mkdir -p $cov_build
	cov-build --dir $cov_build $MAKEP all
	if [ "${ucx_build_type}" == "devel" ]; then
		cov-manage-emit --dir $cov_build --tu-pattern "file('.*/test/gtest/common/googletest/*')" delete
	fi
	cov-analyze --jobs $parallel_jobs $COV_OPT --security --concurrency --dir $cov_build
	nerrors=$(cov-format-errors --dir $cov_build | awk '/Processing [0-9]+ errors?/ { print $2 }')
	rc=$(($rc+$nerrors))

	if [ $nerrors -gt 0 ]; then
		cov-format-errors --dir $cov_build --emacs-style
		cp -ar $cov_build $WORKSPACE/$cov_build_id
		echo "not ok 1 Coverity Detected $nerrors failures"
	else
		echo "ok 1 Coverity found no issues"
		rm -rf $cov_build
	fi
	modules_for_coverity_unload
	return $rc
}

prepare_build
run_coverity "$@"
