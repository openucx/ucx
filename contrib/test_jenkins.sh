#!/bin/bash -eExl
#
# Testing script for OpenUCX, to run from Jenkins CI
#
# Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
# Copyright (C) ARM Ltd. 2016-2018.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#
#
# Environment variables set by Jenkins CI:
#  - WORKSPACE         : path to work dir
#  - BUILD_NUMBER      : jenkins build number
#  - JOB_URL           : jenkins job url
#  - EXECUTOR_NUMBER   : number of executor within the test machine
#  - JENKINS_RUN_TESTS : whether to run unit tests
#  - JENKINS_TEST_PERF : whether to validate performance
#
# Optional environment variables (could be set by job configuration):
#  - nworkers : number of parallel executors
#  - worker   : number of current parallel executor
#  - COV_OPT  : command line options for Coverity static checker
#

WORKSPACE=${WORKSPACE:=$PWD}
ucx_inst=${WORKSPACE}/install

if [ -z "$BUILD_NUMBER" ]; then
	echo "Running interactive"
	BUILD_NUMBER=1
	WS_URL=file://$WORKSPACE
	JENKINS_RUN_TESTS=yes
	JENKINS_TEST_PERF=1
	TIMEOUT=""
	TIMEOUT_VALGRIND=""
else
	echo "Running under jenkins"
	WS_URL=$JOB_URL/ws
	TIMEOUT="timeout 160m"
	TIMEOUT_VALGRIND="timeout 200m"
fi


#
# Set affinity to 2 cores according to Jenkins executor number
#
if [ -n "$EXECUTOR_NUMBER" ] && [ -n "$JENKINS_RUN_TESTS" ]
then
	AFFINITY="taskset -c $(( 2 * EXECUTOR_NUMBER ))","$(( 2 * EXECUTOR_NUMBER + 1))"
else
	AFFINITY=""
fi

#
# Build command runs with 10 tasks
#
MAKE="make"
MAKEP="make -j10"


#
# Set up parallel test execution - "worker" and "nworkers" should be set by jenkins
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
	m_avail="$(module avail $module 2>&1)" || true

	if module avail -t 2>&1 | grep -q "^$module\$"
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
# try load cuda modules if nvidia driver is installed
#
try_load_cuda_env() {
	num_gpus=0
	have_cuda=no
	if [ -f "/proc/driver/nvidia/version" ]; then
		have_cuda=yes
		module_load dev/cuda    || have_cuda=no
		module_load dev/gdrcopy || have_cuda=no
		num_gpus=$(nvidia-smi -L | wc -l)
	fi
}

unload_cuda_env() {
	module unload dev/cuda
	module unload dev/gdrcopy
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
	device_list=$(ibv_devinfo -l | tail -n +2 | sed -e 's/^[ \t]*//' | head -n -1)
	for ibdev in $device_list
	do
		port=1
		(ibv_devinfo -d $ibdev -i $port | grep -q PORT_ACTIVE) && echo "$ibdev:$port" || true
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
	doxy_ready=0
	doxy_target_version="1.8.11"
	doxy_version="$(doxygen --version)" || true

	# Try load newer doxygen if native is older than 1.8.11
	if ! (echo $doxy_target_version; echo $doxy_version) | sort -CV
	then
		if module_load tools/doxygen-1.8.11
		then
			doxy_ready=1
		else
			echo " doxygen was not found"
		fi
	else
		doxy_ready=1
	fi

	if [ $doxy_ready -eq 1 ]
	then
		echo " ==== Build docs only ===="
		../configure --prefix=$ucx_inst --with-docs-only
		$MAKEP clean
		$MAKE  docs
		$MAKEP clean # FIXME distclean does not work with docs-only
	fi
}

#
# Building java docs
#
build_java_docs() {
	echo " ==== Building java docs ===="
	if module_load dev/jdk && module_load dev/mvn
	then
		../configure --prefix=$ucx_inst --with-java
		$MAKE -C ../build-test/bindings/java/src/main/native docs
		module unload dev/jdk
		module unload dev/mvn
	else
		echo "No jdk and mvn module, failed to build docs".
	fi
}

#
# Build without verbs
#
build_no_verbs() {
	echo "==== Build without IB verbs ===="
	../contrib/configure-release --prefix=$ucx_inst --without-verbs
	$MAKEP clean
	$MAKEP
	$MAKEP distclean
}

#
# Build without numa support check
#
build_disable_numa() {
	echo "==== Check --disable-numa compilation option ===="
	../contrib/configure-release --prefix=$ucx_inst --disable-numa
	$MAKEP clean
	$MAKEP
	$MAKEP distclean
}

#
# Build a package in release mode
#
build_release_pkg() {
	echo "==== Build release ===="
	../contrib/configure-release
	$MAKEP clean
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
		../contrib/buildrpm.sh -s -b --nodeps
	fi

	# check that UCX version is present in spec file
	cd ${WORKSPACE}
	# extract version from configure.ac and convert to MAJOR.MINOR.PATCH representation
	version=$(grep -P "define\S+ucx_ver" configure.ac | awk '{print $2}' | sed 's,),,' | xargs echo | tr ' ' '.')
	if ! grep -q "$version" ucx.spec.in; then
		echo "Current UCX version ($version) is not present in ucx.spec.in changelog"
		exit 1
	fi
	cd -

	$MAKEP distclean
}

#
# Build with Intel compiler
#
build_icc() {
	echo 1..1 > build_icc.tap
	if module_load intel/ics && icc -v
	then
		echo "==== Build with Intel compiler ===="
		../contrib/configure-devel --prefix=$ucx_inst CC=icc CXX=icpc
		$MAKEP clean
		$MAKEP
		$MAKEP distclean
		echo "ok 1 - build successful " >> build_icc.tap
	else
		echo "==== Not building with Intel compiler ===="
		echo "ok 1 - # SKIP because Coverity not installed" >> build_icc.tap
	fi
	module unload intel/ics
}

#
# Build debug version
#
build_debug() {
	echo "==== Build with --enable-debug option ===="
	../contrib/configure-devel --prefix=$ucx_inst --enable-debug --enable-examples
	$MAKEP clean
	$MAKEP
	$MAKEP distclean
}

#
# Build UGNI
#
build_ugni() {
    echo 1..1 > build_ugni.tap

    echo "==== Build with cray-ugni ===="
    #
    # Point pkg-config to contrib/cray-ugni-mock, and replace
    # PKG_CONFIG_TOP_BUILD_DIR with source dir, since the mock .pc files contain
    # relative paths.
    #
    ../contrib/configure-devel --prefix=$ucx_inst --with-ugni \
        PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$PWD/../contrib/cray-ugni-mock \
        PKG_CONFIG_TOP_BUILD_DIR=$PWD/..
    $MAKEP clean
    $MAKEP

    # make sure UGNI transport is enabled
    grep '#define HAVE_TL_UGNI 1' config.h

    $MAKE  distcheck
    $MAKEP distclean

    module unload dev/cray-ugni
    echo "ok 1 - build successful " >> build_ugni.tap
}

#
# Build CUDA
#
build_cuda() {
    echo 1..1 > build_cuda.tap
    if module_load dev/cuda
    then
        if module_load dev/gdrcopy
        then
            echo "==== Build with enable cuda, gdr_copy ===="
            ../contrib/configure-devel --prefix=$ucx_inst --with-cuda --with-gdrcopy
            $MAKEP clean
            $MAKEP
            $MAKEP distclean

            ../contrib/configure-release --prefix=$ucx_inst --with-cuda --with-gdrcopy
            $MAKEP clean
            $MAKEP
            $MAKEP distclean
            module unload dev/gdrcopy
        fi

        echo "==== Build with enable cuda, w/o gdr_copy ===="
        ../contrib/configure-devel --prefix=$ucx_inst --with-cuda --without-gdrcopy
        $MAKEP clean
        $MAKEP

        module unload dev/cuda

        echo "==== Running test_link_map with cuda build but no cuda module ===="
        env UCX_HANDLE_ERRORS=bt ./test/apps/test_link_map

        $MAKEP distclean
        echo "ok 1 - build successful " >> build_cuda.tap
    else
        echo "==== Not building with cuda flags ===="
        echo "ok 1 - # SKIP because cuda not installed" >> build_cuda.tap
    fi
}

#
# Build with clang compiler
#
build_clang() {
	echo 1..1 > build_clang.tap
	if which clang > /dev/null 2>&1
	then
		echo "==== Build with clang compiler ===="
		../contrib/configure-devel --prefix=$ucx_inst CC=clang CXX=clang++
		$MAKEP clean
		$MAKEP
		$MAKEP install
		UCX_HANDLE_ERRORS=bt,freeze UCX_LOG_LEVEL_TRIGGER=ERROR $ucx_inst/bin/ucx_info -d
		$MAKEP distclean
		echo "ok 1 - build successful " >> build_clang.tap
	else
		echo "==== Not building with clang compiler ===="
		echo "ok 1 - # SKIP because clang not installed" >> build_clang.tap
	fi
}

#
# Build with gcc-latest module
#
build_gcc_latest() {
	echo 1..1 > build_gcc_latest.tap
    #If the gcc version on the host is older than 4.8.5, don't run
    if (echo "4.8.5"; gcc --version | head -1 | awk '{print $3}') | sort -CV
	then
		if module_load dev/gcc-latest
		then
			echo "==== Build with GCC compiler ($(gcc --version|head -1)) ===="
			../contrib/configure-devel --prefix=$ucx_inst
			$MAKEP clean
			$MAKEP
			$MAKEP install
			UCX_HANDLE_ERRORS=bt,freeze UCX_LOG_LEVEL_TRIGGER=ERROR $ucx_inst/bin/ucx_info -d
			$MAKEP distclean
			echo "ok 1 - build successful " >> build_gcc_latest.tap
		else
			echo "==== Not building with latest gcc compiler ===="
			echo "ok 1 - # SKIP because dev/gcc-latest module is not available" >> build_gcc_latest.tap
		fi
	else
		echo "==== Not building with gcc compiler ===="
		echo "GCC version is too old ($(gcc --version|head -1))"
		echo "ok 1 - # SKIP because gcc version is older than 4.8.5" >> build_gcc_latest.tap
	fi
}

#
# Install and check experimental headers
#
build_experimental_api() {
	# Experimental header file should not be installed by regular build
	echo "==== Install WITHOUT experimental API ===="
	../contrib/configure-release --prefix=$ucx_inst
	$MAKEP clean
	$MAKEP install
	! test -e $ucx_inst/include/ucp/api/ucpx.h

	# Experimental header file should be installed by --enable-experimental-api
	echo "==== Install WITH experimental API ===="
	../contrib/configure-release --prefix=$ucx_inst --enable-experimental-api
	$MAKEP clean
	$MAKEP install
	test -e $ucx_inst/include/ucp/api/ucpx.h
}

#
# Builds jucx
#
build_jucx() {
	echo 1..1 > build_jucx.tap
	if module_load dev/jdk && module_load dev/mvn
	then
		echo "==== Building JUCX bindings (java api for ucx) ===="
		../contrib/configure-release --prefix=$ucx_inst --with-java
		$MAKEP clean
		$MAKEP
		$MAKEP install
		$MAKEP distclean
		echo "ok 1 - build successful " >> build_jucx.tap
		module unload dev/jdk
		module unload dev/mvn
	else
		echo "==== No jdk and mvn modules ==== "
		echo "ok 1 - # SKIP because dev/jdk and dev/mvn modules are not available" >> build_jucx.tap
	fi
}

#
# Build with armclang compiler
#
build_armclang() {
    echo 1..1 > build_armclang.tap
    if module_load arm-compiler/latest
    then
        echo "==== Build with armclang compiler ===="
        ../contrib/configure-devel --prefix=$ucx_inst CC=armclang CXX=armclang++
        $MAKEP clean
        $MAKEP
        $MAKEP install
        UCX_HANDLE_ERRORS=bt,freeze UCX_LOG_LEVEL_TRIGGER=ERROR $ucx_inst/bin/ucx_info -d
        $MAKEP distclean
        echo "ok 1 - build successful " >> build_armclang.tap
        module unload arm-compiler/latest
    else
        echo "==== Not building with armclang compiler ===="
        echo "ok 1 - # SKIP because armclang not installed" >> build_armclang.tap
    fi
}

check_inst_headers() {
	echo 1..1 > inst_headers.tap
	echo "==== Testing installed headers ===="

	../contrib/configure-release --prefix=$PWD/install
	$MAKEP clean
	$MAKEP install
	../contrib/check_inst_headers.sh $PWD/install/include
	$MAKEP distclean

	echo "ok 1 - build successful " >> inst_headers.tap
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

	# set smaller timeouts so the test will complete faster
	if [[ ${test_args} == *"-e"* ]]
	then
		export UCX_UD_TIMEOUT=1s
		export UCX_RC_TIMEOUT=1ms
		export UCX_RC_RETRY_COUNT=4
	fi
	
	# hello-world example
	tcp_port=$((10000 + EXECUTOR_NUMBER))

	./${test_name} ${test_args} -p ${tcp_port} &
	hw_server_pid=$!

	sleep 15

	# temporary disable 
	if [[ ${test_args} == *"-e"* ]]
	then
		set +Ee
	fi

	# need to be ran in background to reflect application PID in $!
	./${test_name} ${test_args} -n $(hostname) -p ${tcp_port} &
	hw_client_pid=$!

	# make sure server process is not running
	if [[ ${test_args} == *"-e"* ]]
	then
		unset UCX_UD_TIMEOUT
		unset UCX_RC_TIMEOUT
		unset UCX_RC_RETRY_COUNT
		wait ${hw_client_pid}
		# return default
		set -Ee
		wait ${hw_server_pid}
	else
		wait ${hw_client_pid} ${hw_server_pid}
	fi
}

#
# Compile and run UCP hello world example
#
run_ucp_hello() {
	if ./src/tools/info/ucx_info -e -u twe|grep ERROR
	then
		return # skip if cannot create ucp ep
	fi

	for test_mode in -w -f -b -e
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
	for send_func in -i -b -z
	do
		for ucx_dev in $(get_active_ib_devices)
		do
			echo "==== Running UCT hello world server on rc/${ucx_dev} with sending ${send_func} ===="
			run_hello uct  -d ${ucx_dev} -t "rc" ${send_func}
		done
		for ucx_dev in $(ip addr | awk '/state UP/ {print $2}' | sed s/://)
		do
			echo "==== Running UCT hello world server on tcp/${ucx_dev} with sending ${send_func} ===="
			run_hello uct -d ${ucx_dev} -t "tcp" ${send_func}
		done
	done
	rm -f ./uct_hello_world
}

run_client_server() {

    test_name=ucp_client_server

    if [ ! -x ${test_name} ]
    then
        gcc -o ${test_name} ${ucx_inst}/share/ucx/examples/${test_name}.c \
            -lucp -lucs -I${ucx_inst}/include -L${ucx_inst}/lib \
            -Wl,-rpath=${ucx_inst}/lib
    fi

    iface=`ibdev2netdev | grep Up | awk '{print $5}' | head -1`
    if [ -n "$iface" ]
    then
        server_ip=`ip addr show ${iface} | awk '/inet /{print $2}' | awk -F '/' '{print $1}'`
    fi

    if [ -z "$server_ip" ]
    then
        # if there is no inet (IPv4) address, bail
        return
    fi

    ibdev=`ibdev2netdev | grep $iface | awk '{print $1}'`
    node_guid=`cat /sys/class/infiniband/$ibdev/node_guid`
    if [ $node_guid == "0000:0000:0000:0000" ]
    then
        return
    fi

    server_port=$((10000 + EXECUTOR_NUMBER))

    # run server side
    ./${test_name} -p ${server_port} &
    hw_server_pid=$!

    sleep 15

    # need to be ran in background to reflect application PID in $!
    ./${test_name} -a ${server_ip} -p ${server_port} &
    hw_client_pid=$!

    wait ${hw_client_pid}
    kill -9 ${hw_server_pid}
}

run_ucp_client_server() {

    if [ ! -r /dev/infiniband/rdma_cm  ]
    then
        return
    fi

    ret=`which ibdev2netdev`
    if [ -z "$ret" ]
    then
        return
    fi

    echo "==== Running UCP client-server  ===="
    run_client_server

    rm -f ./ucp_client_server
}

#
# Run UCX performance test with MPI
#
run_ucx_perftest_mpi() {
	ucx_inst_ptest=$ucx_inst/share/ucx/perftest

	# hack for perftest, no way to override params used in batch
	# todo: fix in perftest
	sed -s 's,-n [0-9]*,-n 1000,g' $ucx_inst_ptest/msg_pow2 | sort -R > $ucx_inst_ptest/msg_pow2_short
	cat $ucx_inst_ptest/test_types_uct |                sort -R > $ucx_inst_ptest/test_types_short_uct
	cat $ucx_inst_ptest/test_types_ucp | grep -v cuda | sort -R > $ucx_inst_ptest/test_types_short_ucp

	UCT_PERFTEST="$ucx_inst/bin/ucx_perftest \
					-b $ucx_inst_ptest/test_types_short_uct \
					-b $ucx_inst_ptest/msg_pow2_short -w 1"

	UCP_PERFTEST="$ucx_inst/bin/ucx_perftest \
					-b $ucx_inst_ptest/test_types_short_ucp \
					-b $ucx_inst_ptest/msg_pow2_short -w 1"

	# shared memory, IB
	devices="posix $(get_active_ib_devices)"

	# Run on all devices
	my_devices=$(get_my_tasks $devices)
	for ucx_dev in $my_devices
	do
		if [[ $ucx_dev =~ .*mlx5.* ]]; then
			opt_transports="-b $ucx_inst_ptest/transports"
			tls=`awk '{print $3 }' $ucx_inst_ptest/transports | tr '\n' ',' | sed -r 's/,$//; s/mlx5/x/g'`
			ucx_env_vars="-x UCX_NET_DEVICES=$ucx_dev -x UCX_TLS=$tls"
		elif [[ $ucx_dev =~ posix ]]; then
			opt_transports="-x mm"
			ucx_env_vars="-x UCX_TLS=mm"
		else
			opt_transports="-x rc"
			ucx_env_vars="-x UCX_NET_DEVICES=$ucx_dev -x UCX_TLS=rc"
		fi

		echo "==== Running ucx_perf kit on $ucx_dev ===="
		$MPIRUN -np 2 $AFFINITY $UCT_PERFTEST -d $ucx_dev $opt_transports
		$MPIRUN -np 2 $ucx_env_vars $AFFINITY $UCP_PERFTEST
	done

	# run cuda tests if cuda module was loaded and GPU is found
	if [ "X$have_cuda" == "Xyes" ] && (lsmod | grep -q "nv_peer_mem") && (lsmod | grep -q "gdrdrv")
	then
		export CUDA_VISIBLE_DEVICES=$(($worker%$num_gpus)),$(($(($worker+1))%$num_gpus))
		cat $ucx_inst_ptest/test_types_ucp | grep cuda | sort -R > $ucx_inst_ptest/test_types_short_ucp
		echo "==== Running ucx_perf with cuda memory===="
		$MPIRUN -np 2 -x UCX_TLS=rc,cuda_copy,gdr_copy -x UCX_MEMTYPE_CACHE=y $AFFINITY $UCP_PERFTEST
		$MPIRUN -np 2 -x UCX_TLS=rc,cuda_copy,gdr_copy -x UCX_MEMTYPE_CACHE=n $AFFINITY $UCP_PERFTEST
		$MPIRUN -np 2 -x UCX_TLS=rc,cuda_copy $AFFINITY $UCP_PERFTEST
		$MPIRUN -np 2 -x UCX_TLS=self,mm,cma,cuda_copy $AFFINITY $UCP_PERFTEST
		$MPIRUN -np 2 $AFFINITY $UCP_PERFTEST
		unset CUDA_VISIBLE_DEVICES
	fi
}

#
# Test malloc hooks with mpi
#
test_malloc_hooks_mpi() {
	for tname in malloc_hooks malloc_hooks_unmapped external_events flag_no_install
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
	if module_load hpcx-gcc && ucx_info -v
	then
		# Prevent our tests from using UCX libraries from hpcx module by prepending
		# our local library path first
		export LD_LIBRARY_PATH=${ucx_inst}/lib:$LD_LIBRARY_PATH

		../contrib/configure-release --prefix=$ucx_inst --with-mpi # TODO check in -devel mode as well
		$MAKEP clean
		$MAKEP install
		$MAKEP installcheck # check whether installation is valid (it compiles examples at least)

		MPIRUN="mpirun \
				-x UCX_ERROR_SIGNALS \
				-x UCX_HANDLE_ERRORS \
				-mca pml ob1 \
				-mca btl tcp,self \
				-mca btl_tcp_if_include lo \
				-mca coll ^hcoll,ml"

		run_ucx_perftest_mpi
		echo "ok 1 - ucx perftest" >> mpi_tests.tap

		test_malloc_hooks_mpi
		echo "ok 2 - malloc hooks" >> mpi_tests.tap

		$MAKEP distclean

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
	echo "==== Running profiling example  ===="

	# configure release mode, application profiling should work
	../contrib/configure-release --prefix=$ucx_inst
	$MAKEP clean
	$MAKEP

	# compile the profiling example code
	gcc -o ucx_profiling ${ucx_inst}/share/ucx/examples/ucx_profiling.c \
		-lm -lucs -I${ucx_inst}/include -L${ucx_inst}/lib -Wl,-rpath=${ucx_inst}/lib

	UCX_PROFILE_MODE=log UCX_PROFILE_FILE=ucx_jenkins.prof ./ucx_profiling

	UCX_READ_PROFILE=${ucx_inst}/bin/ucx_read_profile
	$UCX_READ_PROFILE -r ucx_jenkins.prof | grep "printf" -C 20
	$UCX_READ_PROFILE -r ucx_jenkins.prof | grep -q "calc_pi"
	$UCX_READ_PROFILE -r ucx_jenkins.prof | grep -q "print_pi"
}

test_ucs_load() {
	../contrib/configure-release --prefix=$ucx_inst
	$MAKEP clean
	$MAKEP

	# Make sure UCS library constructor does not call socket()
	echo "==== Running UCS library loading test ===="
	strace ./ucx_profiling &> strace.log
	! grep '^socket' strace.log
}

test_ucs_dlopen() {
	$MAKEP

	# Make sure UCM is not unloaded
	echo "==== Running UCS dlopen test with memhooks ===="
	./test/apps/test_ucs_dlopen
}

test_ucp_dlopen() {
	../contrib/configure-release --prefix=$ucx_inst
	$MAKEP clean
	$MAKEP

	# Make sure UCP library, when opened with dlopen(), loads CMA module
	LIB_CMA=`find ${ucx_inst} -name libuct_cma.so.0`
	if [ -n "$LIB_CMA" ]
	then
		echo "==== Running UCP library loading test ===="
		./test/apps/test_ucp_dlopen # just to save output to log
		./test/apps/test_ucp_dlopen | grep 'cma/cma'
	else
		echo "==== Not running UCP library loading test ===="
	fi
}

test_memtrack() {
	../contrib/configure-devel --prefix=$ucx_inst
	$MAKEP clean
	$MAKEP

	echo "==== Running memtrack test ===="
	UCX_MEMTRACK_DEST=stdout ./test/gtest/gtest --gtest_filter=test_memtrack.sanity
}

test_unused_env_var() {
	# We must create a UCP worker to get the warning about unused variables
	echo "==== Running ucx_info env vars test ===="
	UCX_IB_PORTS=mlx5_0:1 ./src/tools/info/ucx_info -epw -u t | grep "unused" | grep -q "UCX_IB_PORTS"
}

test_env_var_aliases() {
	echo "==== Running MLX5 env var aliases test ===="
	if [[ `./src/tools/info/ucx_info -b | grep -P 'HW_TM *1$'` ]]
	then
		vars=( "TM_ENABLE" "TM_MAX_BCOPY" "TM_LIST_SIZE" "TX_MAX_BB" )
		for var in "${vars[@]}"
		do
			for tl in "RC_MLX5" "DC_MLX5"
			do
				val=$(./src/tools/info/ucx_info -c | grep "${tl}_${var}" | cut -d'=' -f2)
				if [ -z $val ]
				then
					echo "UCX_${tl}_${var} does not exist in UCX config"
					exit 1
				fi
				# To check that changing env var takes an effect,
				# create some value, which is different from the default.
				magic_val=`echo $val | sed -e ' s/inf\|auto/15/; s/n/swap/; s/y/n/; s/swap/y/; s/\([0-9]\)/\11/'`

				# Check that both (tl name and common RC) aliases work
				for var_alias in "RC" $tl
				do
					var_name=UCX_${var_alias}_${var}
					val_set=$(export $var_name=$magic_val; ./src/tools/info/ucx_info -c | grep "${tl}_${var}" | cut -d'=' -f2)
					if [ "$val_set" != "$magic_val" ]
					then
						echo "Can't set $var_name"
						exit 1
					fi
				done
			done
		done
	else
		echo "HW TM is not compiled in UCX"
	fi
}

test_malloc_hook() {
	echo "==== Running malloc hooks test ===="
	if [ -x ./test/apps/test_tcmalloc ]
	then
		./test/apps/test_tcmalloc
	fi
}

test_jucx() {
	echo "==== Running jucx test ===="
	echo "1..2" > jucx_tests.tap
	if module_load dev/jdk && module_load dev/mvn
	then
		jucx_port=$((20000 + EXECUTOR_NUMBER))
		export JUCX_TEST_PORT=jucx_port
		export UCX_ERROR_SIGNALS=""
		$MAKE -C bindings/java/src/main/native test
		unset UCX_ERROR_SIGNALS
		unset JUCX_TEST_PORT
		module unload dev/jdk
		module unload dev/mvn
		echo "ok 1 - jucx test" >> jucx_tests.tap
	else
		echo "Failed to load dev/jdk and dev/mvn modules." >> jucx_tests.tap
	fi
}

#
# Run Coverity and report errors
#
run_coverity() {
	echo 1..1 > coverity.tap
	if module_load tools/cov
	then
		echo "==== Running coverity ===="
		$MAKEP clean
		cov_build_id="cov_build_${BUILD_NUMBER}"
		cov_build="$WORKSPACE/$cov_build_id"
		rm -rf $cov_build
		cov-build   --dir $cov_build $MAKEP all
		cov-analyze $COV_OPT --dir $cov_build
		nerrors=$(cov-format-errors --dir $cov_build | awk '/Processing [0-9]+ errors?/ { print $2 }')
		rc=$(($rc+$nerrors))

		index_html=$(cd $cov_build && find . -name index.html | cut -c 3-)
		if [ -z "$BUILD_URL" ]; then
			cov_url="${WS_URL}/${cov_build_id}/${index_html}"
		else
			cov_url="${BUILD_URL}/artifact/${cov_build_id}/${index_html}"
		fi
		rm -f jenkins_sidelinks.txt
		if [ $nerrors -gt 0 ]; then
			cov-format-errors --dir $cov_build --emacs-style
			echo "not ok 1 Coverity Detected $nerrors failures # $cov_url" >> coverity.tap
		else
			echo "ok 1 Coverity found no issues" >> coverity.tap
			rm -rf $cov_build
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

run_gtest_watchdog_test() {
	watchdog_timeout=$1
	sleep_time=$2
	expected_runtime=$3
	expected_err_str="Connection timed out - abort testing"

	make -C test/gtest

	start_time=`date +%s`

	env WATCHDOG_GTEST_TIMEOUT_=$watchdog_timeout \
		WATCHDOG_GTEST_SLEEP_TIME_=$sleep_time \
		GTEST_FILTER=test_watchdog.watchdog_timeout \
		./test/gtest/gtest 2>&1 | tee watchdog_timeout_test &
	pid=$!
	wait $pid

	end_time=`date +%s`

	res="$(grep -x "$expected_err_str" watchdog_timeout_test)" || true

	rm -f watchdog_timeout_test

	if [ "$res" != "$expected_err_str" ]
	then
		echo "didn't find [$expected_err_str] string in the test output"
		exit 1
	fi

	runtime=$(($end_time-$start_time))

	if [ $runtime -gt $expected_runtime ]
	then
		echo "Watchdog timeout test takes $runtime seconds that" \
			"is greater than expected $expected_runtime seconds"
		exit 1
	fi
}

#
# Run the test suite (gtest)
# Arguments: <compiler-name> [configure-flags]
#
run_gtest() {
	compiler_name=$1
	shift
	../contrib/configure-devel --prefix=$ucx_inst $@
	$MAKEP clean
	$MAKEP

	echo "==== Running watchdog timeout test, $compiler_name compiler ===="
	run_gtest_watchdog_test 5 60 300

	export GTEST_SHARD_INDEX=$worker
	export GTEST_TOTAL_SHARDS=$nworkers
	export GTEST_RANDOM_SEED=0
	export GTEST_SHUFFLE=1
	export GTEST_TAP=2
	export GTEST_REPORT_DIR=$WORKSPACE/reports/tap
	# Run UCT tests for TCP over RDMA devices only
	export GTEST_UCT_TCP_FASTEST_DEV=1

	if [ $num_gpus -gt 0 ]; then
		export CUDA_VISIBLE_DEVICES=$(($worker%$num_gpus))
	fi

	GTEST_EXTRA_ARGS=""
	if [ "$JENKINS_TEST_PERF" == 1 ]
	then
		# Check performance with 10 retries and 2 seconds interval
		GTEST_EXTRA_ARGS="$GTEST_EXTRA_ARGS -p 10 -i 2.0"
	fi
	export GTEST_EXTRA_ARGS

	mkdir -p $GTEST_REPORT_DIR

	echo "==== Running unit tests, $compiler_name compiler ===="
	$AFFINITY $TIMEOUT make -C test/gtest test
	(cd test/gtest && rename .tap _gtest.tap *.tap && mv *.tap $GTEST_REPORT_DIR)

	echo "==== Running malloc hooks mallopt() test, $compiler_name compiler ===="
	# gtest returns with non zero exit code if there were no
	# tests to run. As a workaround run a single test on every
	# shard.
	$AFFINITY $TIMEOUT \
		env UCX_IB_RCACHE=n \
		MALLOC_TRIM_THRESHOLD_=-1 \
		MALLOC_MMAP_THRESHOLD_=-1 \
		GTEST_SHARD_INDEX=0 \
		GTEST_TOTAL_SHARDS=1 \
		GTEST_FILTER=malloc_hook_cplusplus.mallopt \
		make -C test/gtest test
	(cd test/gtest && rename .tap _mallopt_gtest.tap malloc_hook_cplusplus.tap && mv *.tap $GTEST_REPORT_DIR)

	echo "==== Running malloc hooks mmap_ptrs test with MMAP_THRESHOLD=16384, $compiler_name compiler ===="
	$AFFINITY $TIMEOUT \
		env MALLOC_MMAP_THRESHOLD_=16384 \
		GTEST_SHARD_INDEX=0 \
		GTEST_TOTAL_SHARDS=1 \
		GTEST_FILTER=malloc_hook_cplusplus.mmap_ptrs \
		make -C test/gtest test
	(cd test/gtest && rename .tap _mmap_ptrs_gtest.tap malloc_hook_cplusplus.tap && mv *.tap $GTEST_REPORT_DIR)

	if ! [[ $(uname -m) =~ "aarch" ]] && ! [[ $(uname -m) =~ "ppc" ]]
	then
		echo "==== Running valgrind tests, $compiler_name compiler ===="

		# Load newer valgrind if naative is older than 3.10
		if ! (echo "valgrind-3.10.0"; valgrind --version) | sort -CV
		then
			module load tools/valgrind-latest
		fi

		export VALGRIND_EXTRA_ARGS="--xml=yes --xml-file=valgrind.xml --child-silent-after-fork=yes --gen-suppressions=all"
		$AFFINITY $TIMEOUT_VALGRIND make -C test/gtest test_valgrind
		(cd test/gtest && rename .tap _vg.tap *.tap && mv *.tap $GTEST_REPORT_DIR)
		module unload tools/valgrind-latest
	else
		echo "==== Not running valgrind tests with $compiler_name compiler ===="
		echo "1..1"                                          > vg_skipped.tap
		echo "ok 1 - # SKIP because running on $(uname -m)" >> vg_skipped.tap
	fi

	unset GTEST_UCT_TCP_FASTEST_DEV
	unset GTEST_SHARD_INDEX
	unset GTEST_TOTAL_SHARDS
	unset GTEST_RANDOM_SEED
	unset GTEST_SHUFFLE
	unset GTEST_TAP
	unset GTEST_REPORT_DIR
	unset GTEST_EXTRA_ARGS
	unset VALGRIND_EXTRA_ARGS
	unset CUDA_VISIBLE_DEVICES
}

run_gtest_default() {
	run_gtest "default"
}

run_gtest_armclang() {
	if module_load arm-compiler/arm-hpc-compiler && armclang -v
	then
		run_gtest "armclang" CC=armclang CXX=armclang++
	else
		echo "==== Not running with armclang compiler ===="
		echo "1..1"                                          > armclang_skipped.tap
		echo "ok 1 - # SKIP because armclang not found"     >> armclang_skipped.tap
	fi
	module unload arm-compiler/arm-hpc-compiler
}


#
# Run the test suite (gtest) in release configuration
#
run_gtest_release() {

	echo "1..1" > gtest_release.tap

	../contrib/configure-release --prefix=$ucx_inst --enable-gtest
	$MAKEP clean
	$MAKEP

	export GTEST_SHARD_INDEX=0
	export GTEST_TOTAL_SHARDS=1
	export GTEST_RANDOM_SEED=0
	export GTEST_SHUFFLE=1
	export GTEST_TAP=2
	export GTEST_REPORT_DIR=$WORKSPACE/reports/tap

	echo "==== Running unit tests (release configuration) ===="
	env GTEST_FILTER=\*test_obj_size\* $AFFINITY $TIMEOUT make -C test/gtest test
	echo "ok 1" >> gtest_release.tap

	unset GTEST_SHARD_INDEX
	unset GTEST_TOTAL_SHARDS
	unset GTEST_RANDOM_SEED
	unset GTEST_SHUFFLE
	unset GTEST_TAP
	unset GTEST_REPORT_DIR
}

run_ucx_tl_check() {

	echo "1..1" > ucx_tl_check.tap

	../test/apps/test_ucx_tls.py $ucx_inst

	if [ $? -ne 0 ]; then
		echo "not ok 1" >> ucx_tl_check.tap
	else
		echo "ok 1" >> ucx_tl_check.tap
	fi
}

#
# Run all tests
#
run_tests() {
	export UCX_HANDLE_ERRORS=freeze,bt
	export UCX_ERROR_SIGNALS=SIGILL,SIGSEGV,SIGBUS,SIGFPE,SIGPIPE,SIGABRT
	export UCX_ERROR_MAIL_TO=$ghprbActualCommitAuthorEmail
	export UCX_ERROR_MAIL_FOOTER=$JOB_URL/$BUILD_NUMBER/console

	do_distributed_task 0 4 build_icc
	do_distributed_task 1 4 build_debug
	do_distributed_task 1 4 build_ugni
	do_distributed_task 2 4 build_cuda
	do_distributed_task 3 4 build_clang
	do_distributed_task 0 4 build_armclang
	do_distributed_task 1 4 build_gcc_latest
	do_distributed_task 2 4 build_experimental_api
	do_distributed_task 0 4 build_jucx

	# all are running mpi tests
	run_mpi_tests

	if module_load dev/jdk && module_load dev/mvn
	then
		../contrib/configure-devel --prefix=$ucx_inst --with-java
	else
		../contrib/configure-devel --prefix=$ucx_inst
	fi
	$MAKEP
	$MAKEP install

	run_ucx_tl_check

	do_distributed_task 1 4 run_ucp_hello
	do_distributed_task 2 4 run_uct_hello
	do_distributed_task 1 4 run_ucp_client_server
	do_distributed_task 3 4 test_profiling
	do_distributed_task 0 4 test_ucp_dlopen
	do_distributed_task 1 4 test_ucs_dlopen
	do_distributed_task 3 4 test_ucs_load
	do_distributed_task 3 4 test_memtrack
	do_distributed_task 0 4 test_unused_env_var
	do_distributed_task 2 4 test_env_var_aliases
	do_distributed_task 1 3 test_malloc_hook
	do_distributed_task 0 3 test_jucx

	# all are running gtest
	run_gtest_default
	run_gtest_armclang

	do_distributed_task 3 4 run_coverity
	do_distributed_task 0 4 run_gtest_release
}

prepare
try_load_cuda_env
do_distributed_task 0 4 build_docs
do_distributed_task 0 4 build_java_docs
do_distributed_task 0 4 build_disable_numa
do_distributed_task 1 4 build_no_verbs
do_distributed_task 2 4 build_release_pkg
do_distributed_task 3 4 check_inst_headers

if [ -n "$JENKINS_RUN_TESTS" ]
then
	run_tests
fi
