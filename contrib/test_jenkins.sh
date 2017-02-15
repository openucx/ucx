#!/bin/bash -eExl
#
# Testing script for OpenUCX, to run from Jenkins CI
#
# Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

WORKSPACE=${WORKSPACE:=$PWD}
MAKE="make -j$(($(nproc) / 2 + 1))"
ucx_inst=${WORKSPACE}/install

if [ -z "$BUILD_NUMBER" ]; then
	echo "Running interactive"
	BUILD_NUMBER=1
	WS_URL=file://$WORKSPACE
	JENKINS_RUN_TESTS=yes
else
	echo "Running under jenkins"
	WS_URL=$JOB_URL/ws
fi

#
# Set up parallel test execution - "worker" and "nworkers" should be set by jenki
#
if [ -z "$worker" ] || [ -z "$nworkers" ]
then
	worker=0
	nworkers=1
fi
echo "==== Running on $(hostname), worker $worker / $nworkers ===="


#
# Test if an environment module exists and load it if yes.
# Otherwise, return error code.
#
module_load() {
	set +x
	module=$1
	if [ -n "$(module avail $module 2>&1)" ]
	then
		module load $module
		set -x
		return 0
	else
		set -x
		return 1
	fi
}

#
# Check whether this test should do a task with given index,
# according to the parallel test execution parameters.
#
should_do_task() {
	set +x
	task=$1
	ntasks=$2
	tasks_per_worker=$(( (ntasks + nworkers - 1) / nworkers ))
	my_tasks_begin=$((tasks_per_worker * worker))
	my_tasks_end=$((my_tasks_begin + tasks_per_worker))

	# set return value to 0 (success) iff ($my_tasks_begin <= $task < $my_tasks_end)
	[ $task -ge $my_tasks_begin ] && [ $task -lt $my_tasks_end ]
	rc=$?
	set -x
	return $rc
}

#
# Do a given task only if the current worker is supposed to do it.
#
do_distributed_task() {
	set +x
	task=$1
	ntasks=$2
	shift 2
	if should_do_task $task $ntasks
	then
		echo "==== Running '$@' (task $task/$ntasks) ===="
		set -x
		$@
	else
		echo "==== Skipping '$@' (task $task/$ntasks) ===="
		set -x
	fi
}

#
# Take a list of tasks, and return only the ones this worker should do
#
get_my_tasks() {
	task_list=$@
	ntasks=$(echo $task_list|wc -w)
	task=0
	my_task_list=""
	for item in $task_list
	do
		should_do_task $task $ntasks && my_task_list="$my_task_list $item"
		task=$((task + 1))
	done
	echo $my_task_list
}

#
# Get list of active IB devices
#
get_active_ib_devices() {
	for ibdev in $(ibstat -l)
	do
		port=1
		(ibstat $ibdev $port | grep -q Active) && echo "$ibdev:$port"
	done
}

#
# Prepare build environment
#
prepare() {
	echo " ==== Prepare ===="
	env
	cd ${WORKSPACE}
	./autogen.sh
	rm -rf build-test
	mkdir -p build-test
	cd build-test
}

#
# Build documentation
#
build_docs() {
	echo " ==== Build docs only ===="
	../configure --prefix=$ucx_inst --with-docs-only
	$MAKE clean
	$MAKE docs
	$MAKE clean # FIXME distclean does not work with docs-only
}

#
# Build without verbs
#
build_no_verbs() {
	echo "==== Build without IB verbs ===="
	../contrib/configure-release --prefix=$ucx_inst --without-verbs
	$MAKE clean
	$MAKE
	$MAKE distclean
}

#
# Build without numa support check
#
build_disable_numa() {
	echo "==== Check --disable-numa compilation option ===="
	../contrib/configure-release --prefix=$ucx_inst --disable-numa
	$MAKE clean
	$MAKE
	$MAKE distclean
}

#
# Build a package in release mode
#
build_release_pkg() {
	echo "==== Build release ===="
	../contrib/configure-release
	$MAKE clean
	$MAKE
	$MAKE distcheck

	# Show UCX info
	./src/tools/info/ucx_info -f -c -v -y -d -b -p -w -e -uart

	if [ -x /usr/bin/dpkg-buildpackage ]; then
		echo "==== Build debian package ===="
		dpkg-buildpackage -us -uc
	else
		echo "==== Build RPM ===="
		../contrib/buildrpm.sh -s -b
	fi

	$MAKE distclean
}

#
# Build with Intel compiler
#
build_icc() {
	echo 1..1 > build_icc.tap
	if module_load intel/ics
	then
		echo "==== Build with Intel compiler ===="
		../contrib/configure-devel --prefix=$ucx_inst CC=icc CXX=icpc
		$MAKE clean
		$MAKE
		$MAKE distclean
		module unload intel/ics
		echo "ok 1 - build successful " >> build_icc.tap
	else
		echo "==== Not building with Intel compiler ===="
		echo "ok 1 - # SKIP because Coverity not installed" >> build_icc.tap
	fi
}

run_hello() {
	api=$1
	shift
	test_args="$@"
	test_name=${api}_hello_world

	if [ ! -x ${test_name} ]
	then
		gcc -o ${test_name} ${ucx_inst}/share/ucx/examples/${test_name}.c \
		-l${api} -lucs -I${ucx_inst}/include -L${ucx_inst}/lib \
		-Wl,-rpath=${ucx_inst}/lib
	fi

	# hello-world example
	tcp_port=$((10000 + EXECUTOR_NUMBER))

	./${test_name} ${test_args} -p ${tcp_port} &
	hw_server_pid=$!

	sleep 5

	# need to be ran in background to reflect application PID in $!
	./${test_name} ${test_args} -n $(hostname) -p ${tcp_port} &
	hw_client_pid=$!

	# make sure server process is not running
	wait ${hw_server_pid} ${hw_client_pid}
}

#
# Compile and run UCP hello world example
#
run_ucp_hello() {
	for test_mode in -w -f -b
	do
		echo "==== Running UCP hello world with mode ${test_mode} ===="
		run_hello ucp ${test_mode}
	done
	rm -f ./ucp_hello_world
}

#
# Compile and run UCT hello world example
#
run_uct_hello() {
	for ucx_dev in $(get_active_ib_devices)
	do
		for send_func in -i -b -z
		do
			echo "==== Running UCT hello world server on ${ucx_dev} with sending ${send_func} ===="
			run_hello uct  -d ${ucx_dev} -t "rc"
		done
	done
	rm -f ./uct_hello_world
}

#
# Run UCX performance test with MPI
#
run_ucx_perftest_mpi() {
	ucx_inst_ptest=$ucx_inst/share/ucx/perftest

	# hack for perftest, no way to override params used in batch
	# todo: fix in perftest
	sed -s 's,-n [0-9]*,-n 1000,g' $ucx_inst_ptest/msg_pow2 | sort -R > $ucx_inst_ptest/msg_pow2_short
	cat $ucx_inst_ptest/test_types | sort -R > $ucx_inst_ptest/test_types_short

	UCX_PERFTEST="$ucx_inst/bin/ucx_perftest \
					-b $ucx_inst_ptest/test_types_short \
					-b $ucx_inst_ptest/msg_pow2_short -w 1"

	# shared memory, IB
	devices="posix $(get_active_ib_devices)"

	# Run on all devices
	my_devices=$(get_my_tasks $devices)
	for ucx_dev in $my_devices
	do
		if [[ $ucx_dev =~ .*mlx5.* ]]; then
			opt_transports="-b $ucx_inst_ptest/transports"
		elif [[ $ucx_dev =~ posix ]]; then
			opt_transports="-x mm"
		else
			opt_transports="-x rc"
		fi

		echo "==== Running ucx_perf kit on $ucx_dev ===="
		$MPIRUN -np 2 $AFFINITY $UCX_PERFTEST -d $ucx_dev $opt_transports
	done
}

#
# Test malloc hooks with mpi
#
test_malloc_hooks_mpi() {
	for tname in malloc_hooks external_events flag_no_install
	do
		echo "==== Running memory hook (${tname}) on MPI ===="
		$MPIRUN -np 1 $AFFINITY ./test/mpi/test_memhooks -t $tname
	done

	echo "==== Running memory hook (malloc_hooks) on MPI with LD_PRELOAD ===="
	ucm_lib=$PWD/src/ucm/.libs/libucm.so
	ls -l $ucm_lib
	$MPIRUN -np 1 -x LD_PRELOAD=$ucm_lib $AFFINITY ./test/mpi/test_memhooks -t malloc_hooks
}

#
# Run tests with MPI library
#
run_mpi_tests() {
	echo "1..2" > mpi_tests.tap
	if module_load hpcx-gcc
	then
		../contrib/configure-release --prefix=$ucx_inst --with-mpi # TODO check in -devel mode as well
		$MAKE clean
		$MAKE install
		$MAKE installcheck # check whether installation is valid (it compiles examples at least)

		# Prevent our tests from using installed UCX libraries
		export LD_LIBRARY_PATH=${ucx_inst}/lib:$LD_LIBRARY_PATH

		MPIRUN="mpirun \
				-x UCX_ERROR_SIGNALS \
				-x UCX_HANDLE_ERRORS \
				-mca pml ob1 \
				-mca btl sm,self \
				-mca coll ^hcoll,ml"

		run_ucx_perftest_mpi
		echo "ok 1 - ucx perftest" >> mpi_tests.tap

		test_malloc_hooks_mpi
		echo "ok 2 - malloc hooks" >> mpi_tests.tap

		$MAKE distclean
		module unload hpcx-gcc
	else
		echo "==== Not running MPI tests ===="
		echo "ok 1 - # SKIP because MPI not installed" >> mpi_tests.tap
		echo "ok 2 - # SKIP because MPI not installed" >> mpi_tests.tap
	fi
}

#
# Test profiling infrastructure
#
test_profiling() {
	echo "==== Running profiling test ===="
	UCX_PROFILE_MODE=log UCX_PROFILE_FILE=ucx_jenkins.prof ./test/apps/test_profiling

	UCX_READ_PROFILE=${ucx_inst}/bin/ucx_read_profile
	$UCX_READ_PROFILE -r ucx_jenkins.prof | grep "printf" -C 20
	$UCX_READ_PROFILE -r ucx_jenkins.prof | grep -q "calc_pi"
	$UCX_READ_PROFILE -r ucx_jenkins.prof | grep -q "print_pi"

	echo "==== Running dlopen test ===="
	strace ./test/apps/test_profiling &> strace.log
	! grep '^socket' strace.log
}

#
# Run Coverity and report errors
#
run_coverity() {
	echo 1..1 > coverity.tap
	if module_load tools/cov
	then
		echo "==== Running coverity ===="
		$MAKE clean
		cov_build_id="cov_build_${BUILD_NUMBER}"
		cov_build="$WORKSPACE/$cov_build_id"
		rm -rf $cov_build
		cov-build   --dir $cov_build $MAKE all
    cov-analyze $COV_OPT --dir $cov_build
		nerrors=$(cov-format-errors --dir $cov_build | awk '/Processing [0-9]+ errors?/ { print $2 }')
		rc=$(($rc+$nerrors))

		index_html=$(cd $cov_build && find . -name index.html | cut -c 3-)
		cov_url="$WS_URL/$cov_build_id/${index_html}"
		rm -f jenkins_sidelinks.txt
		if [ $nerrors -gt 0 ]; then
			cov-format-errors --dir $cov_build --emacs-style
			echo "not ok 1 Coverity Detected $nerrors failures # $cov_url" >> coverity.tap
		else
		echo "ok 1 Coverity found no issues" >> coverity.tap
		fi

		echo Coverity report: $cov_url
		printf "%s\t%s\n" Coverity $cov_url >> jenkins_sidelinks.txt
		module unload tools/cov

		return $rc
	else
		echo "==== Not running Coverity ===="
		echo "ok 1 - # SKIP because Coverity not installed" >> coverity.tap
	fi
}

#
# Run the test suite (gtest)
#
run_gtest() {
	../contrib/configure-devel --prefix=$ucx_inst
	$MAKE clean
	$MAKE

	export GTEST_SHARD_INDEX=$worker
	export GTEST_TOTAL_SHARDS=$nworkers
	export GTEST_RANDOM_SEED=0
	export GTEST_SHUFFLE=1
	export GTEST_TAP=2
	export GTEST_REPORT_DIR=$WORKSPACE/reports/tap

	mkdir -p $GTEST_REPORT_DIR

	echo "==== Running unit tests ===="
	$AFFINITY $TIMEOUT make -C test/gtest test
	(cd test/gtest && rename .tap _gtest.tap *.tap && mv *.tap $GTEST_REPORT_DIR)

	echo "==== Running valgrind tests ===="
	if [ $(valgrind --version) != "valgrind-3.10.0" ]
	then
		module load tools/valgrind-latest
	fi
	export VALGRIND_EXTRA_ARGS="--xml=yes --xml-file=valgrind.xml --child-silent-after-fork=yes"
	$AFFINITY $TIMEOUT_VALGRIND make -C test/gtest test_valgrind
	(cd test/gtest && rename .tap _vg.tap *.tap && mv *.tap $GTEST_REPORT_DIR)
	module unload tools/valgrind-latest
}

#
# Run all tests
#
run_tests() {
	# Set CPU affinity to 2 cores, for performance tests
	if [ -n "$EXECUTOR_NUMBER" ]; then
		AFFINITY="taskset -c $(( 2 * EXECUTOR_NUMBER ))","$(( 2 * EXECUTOR_NUMBER + 1))"
		TIMEOUT="timeout 160m"
		TIMEOUT_VALGRIND="timeout 200m"
	else
		AFFINITY=""
		TIMEOUT=""
		TIMEOUT_VALGRIND=""
	fi

	export UCX_HANDLE_ERRORS=freeze,bt
	export UCX_ERROR_SIGNALS=SIGILL,SIGSEGV,SIGBUS,SIGFPE,SIGPIPE,SIGABRT
	export UCX_ERROR_MAIL_TO=$ghprbActualCommitAuthorEmail
	export UCX_ERROR_MAIL_FOOTER=$JOB_URL/$BUILD_NUMBER/console

	do_distributed_task 0 4 build_icc

	# all are running mpi tests
	run_mpi_tests

    ../contrib/configure-devel --prefix=$ucx_inst
    $MAKE
    $MAKE install

	do_distributed_task 1 4 run_ucp_hello
	do_distributed_task 2 4 run_uct_hello
	do_distributed_task 3 4 test_profiling

	# all are running gtest
	run_gtest

	do_distributed_task 3 4 run_coverity
}

prepare
do_distributed_task 0 4 build_docs
do_distributed_task 0 4 build_disable_numa
do_distributed_task 1 4 build_no_verbs
do_distributed_task 2 4 build_release_pkg

if [ -n "$JENKINS_RUN_TESTS" ]
then
	run_tests
fi
