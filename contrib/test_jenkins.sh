#!/bin/bash -eEx
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
#  - WORKSPACE           : path to work dir
#  - BUILD_NUMBER        : jenkins build number
#  - JOB_URL             : jenkins job url
#  - EXECUTOR_NUMBER     : number of executor within the test machine
#  - JENKINS_RUN_TESTS   : whether to run unit tests
#  - RUN_TESTS           : same as JENKINS_RUN_TESTS, but for Azure
#  - JENKINS_TEST_PERF   : whether to validate performance
#  - JENKINS_NO_VALGRIND : set this to disable valgrind tests
#
# Optional environment variables (could be set by job configuration):
#  - nworkers : number of parallel executors
#  - worker   : number of current parallel executor
#  - COV_OPT  : command line options for Coverity static checker
#

WORKSPACE=${WORKSPACE:=$PWD}
ucx_inst=${WORKSPACE}/install
CUDA_MODULE="dev/cuda11.4"
GDRCOPY_MODULE="dev/gdrcopy2.3_cuda11.4"

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
	TIMEOUT="timeout 200m"
	TIMEOUT_VALGRIND="timeout 240m"
fi


#
# Set affinity to 2 cores according to Jenkins executor number.
# Affinity is inherited from agent in Azure CI.
# TODO: remove or rename after CI migration.
#
if [ -n "$EXECUTOR_NUMBER" ] && [ -n "$JENKINS_RUN_TESTS" ]
then
	AFFINITY="taskset -c $(( 2 * EXECUTOR_NUMBER ))","$(( 2 * EXECUTOR_NUMBER + 1))"
else
	AFFINITY=""
fi

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

have_ptrace=$(capsh --print | grep 'Bounding' | grep ptrace || true)

#
# Set initial port number for client/server applications
#
server_port_range=1000
server_port_min=$((10500 + EXECUTOR_NUMBER * server_port_range))
server_port_max=$((server_port_min + server_port_range))
server_port=${server_port_min}

#
# Override maven repository path, to cache the downloaded packages accross tests
#
export maven_repo=${WORKSPACE}/.deps

#
# Set up parallel test execution - "worker" and "nworkers" should be set by jenkins
#
if [ -z "$worker" ] || [ -z "$nworkers" ]
then
	worker=0
	nworkers=1
fi
echo "==== Running on $(hostname), worker $worker / $nworkers ===="

# Report an warning message to Azure pipeline
log_warning() {
	msg=$1
	test "x$RUNNING_IN_AZURE" = "xyes" && { azure_log_warning "${msg}" ; set -x; } || echo "${msg}"
}

# Report an error message to Azure pipeline
log_error() {
	msg=$1
	test "x$RUNNING_IN_AZURE" = "xyes" && { azure_log_error "${msg}" ; set -x; } || echo "${msg}"
}

#
# cleanup ucx
#
make_clean() {
	rm -rf ${ucx_inst}
	$MAKEP ${1:-clean}
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
		config_args+="--with-iodemo-cuda"
	fi

	../contrib/configure-${mode} ${config_args} "$@"
	make_clean
	$MAKEP
	$MAKEP install
}

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
# Safe unload for env modules (even if it doesn't exist)
#
module_unload() {
	module=$1
	module unload "${module}" || true
}

#
# try load cuda modules if nvidia driver is installed
#
try_load_cuda_env() {
	num_gpus=0
	have_cuda=no
	have_gdrcopy=no
	if [ -f "/proc/driver/nvidia/version" ]; then
		have_cuda=yes
		have_gdrcopy=yes
		module_load $CUDA_MODULE    || have_cuda=no
		module_load $GDRCOPY_MODULE || have_gdrcopy=no
		num_gpus=$(nvidia-smi -L | wc -l)
		export CUDA_VISIBLE_DEVICES=$(($worker%$num_gpus))
	fi
}

unload_cuda_env() {
	module_unload $CUDA_MODULE
	module_unload $GDRCOPY_MODULE
}

#
# Check whether this test should do a task with given index,
# according to the parallel test execution parameters.
#
should_do_task() {
	set +x
	[[ $((task % nworkers)) -eq ${worker} ]]
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
	set +x
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
	set -x
}

#
# Get list IB devices
#
get_ib_devices() {
	state=$1
	device_list=$(ibv_devinfo -l | tail -n +2)
	set +x
	for ibdev in $device_list
	do
		num_ports=$(ibv_devinfo -d $ibdev| awk '/phys_port_cnt:/ {print $2}')
		for port in $(seq 1 $num_ports)
		do
			if [ -e "/sys/class/infiniband/${ibdev}/ports/${port}/gids/0" ] && \
			   ibv_devinfo -d $ibdev -i $port | grep -q $state
			then
				echo "$ibdev:$port"
			fi
		done
	done
	set -x
}

#
# Get IB devices on state Active
#
get_active_ib_devices() {
	get_ib_devices PORT_ACTIVE
}

#
# Check IB devices on state INIT
#
check_machine() {
	init_dev=$(get_ib_devices PORT_INIT)
	if [ -n "${init_dev}" ]
	then
		echo "${init_dev} have state PORT_INIT"
		exit 1
	fi
}

#
# Get list of active IP interfaces
#
get_active_ip_ifaces() {
	device_list=$(ip addr | awk '/state UP/ {print $2}' | sed s/:// | cut -f 1 -d '@')
	for netdev in ${device_list}
	do
		(ip addr show ${netdev} | grep -q 'inet ') && echo ${netdev} || true
	done
}

#
# Get IP addr for a given IP iface
# Argument is the IP iface
#
get_ifaddr() {
	iface=$1
	echo $(ip addr show ${iface} | awk '/inet /{print $2}' | awk -F '/' '{print $1}')
}

get_rdma_device_ip_addr() {
	if [ ! -r /dev/infiniband/rdma_cm  ]
	then
		return
	fi

	if ! which ibdev2netdev >&/dev/null
	then
		return
	fi

	set +x
	ibdev2netdev | grep Up | while read line
	do
		ibdev=$(echo "${line}" | awk '{print $1}')
		port=$(echo "${line}" | awk '{print $3}')
		netif=$(echo "${line}" | awk '{print $5}')
		node_guid=`cat /sys/class/infiniband/${ibdev}/node_guid`

		# skip devices that do not have proper gid (representors)
		if [ -e "/sys/class/infiniband/${ibdev}/ports/${port}/gids/0" ] && \
			[ ${node_guid} != "0000:0000:0000:0000" ]
		then
			get_ifaddr ${netif}
			set -x
			return
		fi
	done
	set -x
}

get_non_rdma_ip_addr() {
	if ! which ibdev2netdev >&/dev/null
	then
		return
	fi

	# get the interface of the ip address that is the default gateway (pure Ethernet IPv4 address).
	eth_iface=$(ip route show| sed -n 's/default via \(\S*\) dev \(\S*\).*/\2/p')

	# the pure Ethernet interface should not appear in the ibdev2netdev output. it should not be an IPoIB or
	# RoCE interface.
	if ibdev2netdev|grep -qw "${eth_iface}"
	then
		echo "Failed to retrieve an IP of a non IPoIB/RoCE interface"
		exit 1
	fi

	get_ifaddr ${eth_iface}
}

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
# Expands a CPU list such as "0-3,17" to "0 1 2 3 17" (each cpu in a new line)
#
expand_cpulist() {
	cpulist=$1
	tokens=$(echo ${cpulist} | tr ',' ' ')
	for token in ${tokens}
	do
		# if there is no '-', first and last would be equal
		first=$(echo ${token} | cut -d'-' -f1)
		last=$( echo ${token} | cut -d'-' -f2)

		for ((cpu=${first};cpu<=${last};++cpu))
		do
			echo ${cpu}
		done
	done
}

#
# Get the N'th CPU that the current process can run on
#
slice_affinity() {
	set +x
	n=$1

	# get affinity mask of the current process
	compact_cpulist=$($AFFINITY bash -c 'taskset -cp $$' | cut -d: -f2)
	cpulist=$(expand_cpulist ${compact_cpulist})

	echo "${cpulist}" | head -n $((n + 1)) | tail -1
	set -x
}

run_loopback_app() {
	test_exe=$1
	test_args="-l $2"

	affinity=$(slice_affinity 0)

	taskset -c $affinity ${test_exe} ${test_args} &
	pid=$!

	wait ${pid} || true
}

step_server_port() {
	# Cycle server_port between (server_port_min)..(server_port_max-1)
	server_port=$((server_port + 1))
	server_port=$((server_port >= server_port_max ? server_port_min : server_port))
}

run_client_server_app() {
	test_exe=$1
	test_args=$2
	server_addr_arg=$3
	kill_server=$4
	error_emulation=$5

	server_port_arg="-p $server_port"
	step_server_port

	affinity_server=$(slice_affinity 0)
	affinity_client=$(slice_affinity 1)

	taskset -c $affinity_server ${test_exe} ${test_args} ${server_port_arg} &
	server_pid=$!

	sleep 15

	if [ $error_emulation -eq 1 ]
	then
		set +Ee
	fi

	taskset -c $affinity_client ${test_exe} ${test_args} ${server_addr_arg} ${server_port_arg} &
	client_pid=$!

	wait ${client_pid}

	if [ $error_emulation -eq 1 ]
	then
		set -eE
	fi

	if [ $kill_server -eq 1 ]
	then
		kill -9 ${server_pid}
	fi
	wait ${server_pid} || true
}

run_hello() {
	api=$1
	shift
	if [[ $1 == "proto" ]]
	then
		export UCX_PROTO_ENABLE=y
		shift
	fi

	test_args="$@"
	test_name=${api}_hello_world

	if [ ! -x ${test_name} ]
	then
		$MAKEP -C examples ${test_name}
	fi

	# set smaller timeouts so the test will complete faster
	if [[ ${test_args} =~ "-e" ]]
	then
		export UCX_UD_TIMEOUT=15s
		export UCX_RC_TIMEOUT=1ms
		export UCX_RC_RETRY_COUNT=4
	fi

	if [[ ${test_args} =~ "-e" ]]
	then
		error_emulation=1
	else
		error_emulation=0
	fi

	run_client_server_app "./examples/${test_name}" "${test_args}" "-n $(hostname)" 0 ${error_emulation}

	if [[ ${test_args} == *"-e"* ]]
	then
		unset UCX_UD_TIMEOUT
		unset UCX_RC_TIMEOUT
		unset UCX_RC_RETRY_COUNT
	fi
	unset UCX_PROTO_ENABLE
}

#
# Compile and run UCP hello world example
#
run_ucp_hello() {
	if ./src/tools/info/ucx_info -e -u twe|grep ERROR
	then
		return # skip if cannot create ucp ep
	fi

	mem_types_list="host "

	if [ "X$have_cuda" == "Xyes" ]
	then
		mem_types_list+="cuda cuda-managed "
	fi

	export UCX_KEEPALIVE_INTERVAL=1s
	export UCX_KEEPALIVE_NUM_EPS=10
	export UCX_LOG_LEVEL=info
	export UCX_MM_ERROR_HANDLING=y

	for tls in all tcp,cuda shm,cuda
	do
		export UCX_TLS=${tls}
		for test_mode in -w -f -b -erecv -esend -ekeepalive proto
		do
			for mem_type in $mem_types_list
			do
				echo "==== Running UCP hello world with mode ${test_mode} and \"${mem_type}\" memory type ===="
				run_hello ucp ${test_mode} -m ${mem_type}
			done
		done
	done
	rm -f ./ucp_hello_world

	unset UCX_KEEPALIVE_INTERVAL
	unset UCX_KEEPALIVE_NUM_EPS
	unset UCX_LOG_LEVEL
	unset UCX_TLS
	unset UCX_MM_ERROR_HANDLING
}

#
# Compile and run UCT hello world example
#
run_uct_hello() {
	mem_types_list="host "

	if [ "X$have_cuda" == "Xyes" ] && [ -f "/sys/kernel/mm/memory_peers/nv_mem/version" ]
	then
		mem_types_list+="cuda-managed "
		if [ -f "/sys/kernel/mm/memory_peers/nv_mem/version" ]
		then
			# test RDMA GPUDirect
			mem_types_list+="cuda "
		fi
	fi

	for send_func in -i -b -z
	do
		for ucx_dev in $(get_active_ib_devices)
		do
			for mem_type in $mem_types_list
			do
				echo "==== Running UCT hello world server on rc/${ucx_dev} with sending ${send_func} and \"${mem_type}\" memory type ===="
				run_hello uct -d ${ucx_dev} -t "rc_verbs" ${send_func} -m ${mem_type}
			done
		done
		for ucx_dev in $(get_active_ip_ifaces)
		do
			echo "==== Running UCT hello world server on tcp/${ucx_dev} with sending ${send_func} ===="
			run_hello uct -d ${ucx_dev} -t "tcp" ${send_func}
		done
	done
	rm -f ./uct_hello_world
}

run_client_server() {
	test_name=ucp_client_server

	mem_types_list="host"

	if [ "X$have_cuda" == "Xyes" ]
	then
		mem_types_list+=" cuda cuda-managed "
	fi

	if [ ! -x ${test_name} ]
	then
		$MAKEP -C examples ${test_name}
	fi

	server_ip=$1
	if [ "$server_ip" == "" ]
	then
		return
	fi

	for mem_type in ${mem_types_list}
	do
		echo "==== Running UCP client-server with \"${mem_type}\" memory type ===="
		run_client_server_app "./examples/${test_name}" "-m ${mem_type}" "-a ${server_ip}" 1 0
	done
}

run_ucp_client_server() {
	echo "==== Running UCP client-server  ===="
	run_client_server $(get_rdma_device_ip_addr)
	run_client_server $(get_non_rdma_ip_addr)
	run_client_server "127.0.0.1"
}

run_io_demo() {
	server_rdma_addr=$(get_rdma_device_ip_addr)
	server_nonrdma_addr=$(get_non_rdma_ip_addr)
	mem_types_list="host "

	if [ "X$have_cuda" == "Xyes" ]
	then
		mem_types_list+="cuda cuda-managed "
	fi

	if [ -z "$server_rdma_addr" ] && [ -z "$server_nonrdma_addr" ]
	then
		return
	fi

	for mem_type in $mem_types_list
	do
		echo "==== Running UCP IO demo with \"${mem_type}\" memory type ===="

		test_args="$@ -o write,read -d 128:4194304 -P 2 -i 10000 -w 10 -c 5 -m ${mem_type} -q"
		test_name=io_demo

		for server_ip in $server_rdma_addr $server_nonrdma_addr
		do
			export UCX_PROTO_ENABLE=y
			run_client_server_app "./test/apps/iodemo/${test_name}" "${test_args}" "${server_ip}" 1 0
			unset UCX_PROTO_ENABLE
			run_client_server_app "./test/apps/iodemo/${test_name}" "${test_args}" "${server_ip}" 1 0
		done

		if [ "${mem_type}" == "host" ]
		then
			run_client_server_app "./test/apps/iodemo/${test_name}" "${test_args}" "127.0.0.1" 1 0
		fi
	done
}

#
# Run UCX performance test
# Note: If requested running with MPI, MPI has to be initialized before
# The function accepts 0 (default value) or 1 that means launching w/ or w/o MPI
#
run_ucx_perftest() {
	if [ $# -eq 0 ]
	then
		with_mpi=0
	else
		with_mpi=$1
	fi
	ucx_inst_ptest=$ucx_inst/share/ucx/perftest

	# hack for perftest, no way to override params used in batch
	# todo: fix in perftest
	sed -s 's,-n [0-9]*,-n 100,g' $ucx_inst_ptest/msg_pow2 | sort -R >  $ucx_inst_ptest/msg_pow2_short
	cat $ucx_inst_ptest/test_types_uct                     | sort -R >  $ucx_inst_ptest/test_types_short_uct
	cat $ucx_inst_ptest/test_types_ucp     | grep -v cuda  | sort -R >  $ucx_inst_ptest/test_types_short_ucp
	cat $ucx_inst_ptest/test_types_ucp_rma | grep -v cuda  | sort -R >> $ucx_inst_ptest/test_types_short_ucp

	ucx_perftest="$ucx_inst/bin/ucx_perftest"
	uct_test_args="-b $ucx_inst_ptest/test_types_short_uct \
				-b $ucx_inst_ptest/msg_pow2_short -w 1"

	ucp_test_args="-b $ucx_inst_ptest/test_types_short_ucp \
				-b $ucx_inst_ptest/msg_pow2_short -w 1"

	# IP ifaces
	ip_ifaces=$(get_active_ip_ifaces)

	# shared memory, IB devices, IP ifaces
	devices="memory $(get_active_ib_devices) ${ip_ifaces}"

	# Run on all devices
	my_devices=$(get_my_tasks $devices)
	for ucx_dev in $my_devices
	do
		if [[ $ucx_dev =~ .*mlx5.* ]]; then
			opt_transports="-b $ucx_inst_ptest/transports"
			tls=`awk '{print $3 }' $ucx_inst_ptest/transports | tr '\n' ',' | sed -r 's/,$//; s/mlx5/x/g'`
			dev=$ucx_dev
		elif [[ $ucx_dev =~ memory ]]; then
			opt_transports="-x posix"
			tls="shm"
			dev="all"
		elif [[ " ${ip_ifaces[*]} " == *" ${ucx_dev} "* ]]; then
			opt_transports="-x tcp"
			tls="tcp"
			dev=$ucx_dev
		else
			opt_transports="-x rc_verbs"
			tls="rc_v"
			dev=$ucx_dev
		fi

		echo "==== Running ucx_perf kit on $ucx_dev ===="
		if [ $with_mpi -eq 1 ]
		then
			# Run UCP performance test
			$MPIRUN -np 2 -x UCX_NET_DEVICES=$dev -x UCX_TLS=$tls $AFFINITY $ucx_perftest $ucp_test_args

			# Run UCP loopback performance test
			$MPIRUN -np 1 -x UCX_NET_DEVICES=$dev -x UCX_TLS=$tls $AFFINITY $ucx_perftest $ucp_test_args "-l"
		else
			export UCX_NET_DEVICES=$dev
			export UCX_TLS=$tls

			# Run UCT performance test
			run_client_server_app "$ucx_perftest" "$uct_test_args -d ${ucx_dev} ${opt_transports}" \
								"$(hostname)" 0 0

			# Run UCT loopback performance test
			run_loopback_app "$ucx_perftest" "$uct_test_args -d ${ucx_dev} ${opt_transports}"

			# Run UCP performance test
			run_client_server_app "$ucx_perftest" "$ucp_test_args" "$(hostname)" 0 0

			# Run UCP performance test with 2 threads
			run_client_server_app "$ucx_perftest" "$ucp_test_args -T 2" "$(hostname)" 0 0

			# Run UCP loopback performance test
			run_loopback_app "$ucx_perftest" "$ucp_test_args"

			unset UCX_NET_DEVICES
			unset UCX_TLS
		fi
	done

	# run cuda tests if cuda module was loaded and GPU is found, and only in
	# client/server mode, to reduce testing time
	if [ "X$have_cuda" == "Xyes" ] && [ $with_mpi -ne 1 ]
	then
		gdr_options="n "
		if (lsmod | grep -q "nv_peer_mem")
		then
			echo "GPUDirectRDMA module (nv_peer_mem) is present.."
			gdr_options+="y "
		fi

		if [ $num_gpus -gt 1 ]; then
			export CUDA_VISIBLE_DEVICES=$(($worker%$num_gpus)),$(($(($worker+1))%$num_gpus))
		fi

		cat $ucx_inst_ptest/test_types_ucp | grep cuda | sort -R > $ucx_inst_ptest/test_types_short_ucp
		sed -s 's,-n [0-9]*,-n 10 -w 1,g' $ucx_inst_ptest/msg_pow2 | sort -R > $ucx_inst_ptest/msg_pow2_short

		echo "==== Running ucx_perf with cuda memory ===="

		for memtype_cache in y n
		do
			for gdr in $gdr_options
			do
				export UCX_MEMTYPE_CACHE=$memtype_cache
				export UCX_IB_GPU_DIRECT_RDMA=$gdr
				run_client_server_app "$ucx_perftest" "$ucp_test_args" "$(hostname)" 0 0
				unset UCX_MEMTYPE_CACHE
				unset UCX_IB_GPU_DIRECT_RDMA
			done
		done

		export UCX_TLS=self,shm,cma,cuda_copy
		run_client_server_app "$ucx_perftest" "$ucp_test_args" "$(hostname)" 0 0
		unset UCX_TLS

		# Specifically test cuda_ipc for large message sizes
		cat $ucx_inst_ptest/test_types_ucp | grep -v cuda | sort -R > $ucx_inst_ptest/test_types_cuda_ucp
		ucp_test_args_large="-b $ucx_inst_ptest/test_types_cuda_ucp \
			             -b $ucx_inst_ptest/msg_pow2_large -w 1"
		for ipc_cache in y n
		do
			export UCX_TLS=self,sm,cuda_copy,cuda_ipc
			export UCX_CUDA_IPC_CACHE=$ipc_cache
			run_client_server_app "$ucx_perftest" "$ucp_test_args_large" "$(hostname)" 0 0
			unset UCX_CUDA_IPC_CACHE
			unset UCX_TLS
		done

		echo "==== Running ucx_perf with cuda memory and new protocols ===="

		# Add RMA tests to the list of tests
		cat $ucx_inst_ptest/test_types_ucp_rma | grep cuda | sort -R >> $ucx_inst_ptest/test_types_short_ucp

		export UCX_PROTO_ENABLE=y
		run_client_server_app "$ucx_perftest" "$ucp_test_args" "$(hostname)" 0 0
		unset UCX_PROTO_ENABLE

		unset CUDA_VISIBLE_DEVICES
	fi
}

#
# Test malloc hooks with mpi
#
test_malloc_hooks_mpi() {
	for mode in reloc bistro
	do
		for tname in malloc_hooks malloc_hooks_unmapped external_events flag_no_install
		do
			echo "==== Running memory hook (${tname} mode ${mode}) on MPI ===="
			$MPIRUN -np 1 $AFFINITY \
				./test/mpi/test_memhooks -t $tname -m ${mode}
		done

		echo "==== Running memory hook (malloc_hooks mode ${mode}) on MPI with LD_PRELOAD ===="
		ucm_lib=$PWD/src/ucm/.libs/libucm.so
		ls -l $ucm_lib
		$MPIRUN -np 1 -x LD_PRELOAD=$ucm_lib $AFFINITY \
			./test/mpi/test_memhooks -t malloc_hooks -m ${mode}
	done
}

#
# Run tests with MPI library
#
run_mpi_tests() {
	prev_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
	mpi_module=hpcx-ga-gcc
	if module_load ${mpi_module}
	then
		if mpirun --version
		then
			# Prevent our tests from using UCX libraries from hpcx module by prepending
			# our local library path first
			save_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
			export LD_LIBRARY_PATH=${ucx_inst}/lib:${MPI_HOME}/lib:${prev_LD_LIBRARY_PATH}

			build release --disable-gtest --with-mpi

			# check whether installation is valid (it compiles examples at least)
			$MAKEP installcheck

			MPIRUN="mpirun \
					--bind-to none \
					-x UCX_ERROR_SIGNALS \
					-x UCX_HANDLE_ERRORS \
					-mca pml ob1 \
					-mca btl tcp,self \
					-mca btl_tcp_if_include lo \
					-mca orte_allowed_exit_without_sync 1 \
					-mca coll ^hcoll,ml"

			run_ucx_perftest 1

			test_malloc_hooks_mpi

			# Restore LD_LIBRARY_PATH so subsequent tests will not take UCX libs
			# from installation directory
			export LD_LIBRARY_PATH=${save_LD_LIBRARY_PATH}

			make_clean distclean
		else
			echo "==== Not running MPI tests ===="
		fi

		module unload ${mpi_module}
	else
		echo "==== Not running MPI tests ===="
	fi
}

build_ucx_profiling_test() {
	# compile the profiling example code
	gcc -o ucx_profiling ../test/apps/profiling/ucx_profiling.c \
		-lm -lucs -I${ucx_inst}/include -L${ucx_inst}/lib -Wl,-rpath=${ucx_inst}/lib
}

#
# Test profiling infrastructure
#
test_profiling() {
	echo "==== Running profiling example  ===="

	build_ucx_profiling_test

	UCX_PROFILE_MODE=log UCX_PROFILE_FILE=ucx_jenkins.prof ./ucx_profiling

	UCX_READ_PROFILE=${ucx_inst}/bin/ucx_read_profile
	$UCX_READ_PROFILE -r ucx_jenkins.prof | grep "printf" -C 20
	$UCX_READ_PROFILE -r ucx_jenkins.prof | grep -q "calc_pi"
	$UCX_READ_PROFILE -r ucx_jenkins.prof | grep -q "print_pi"
}

test_ucs_load() {
	if [ -z "${have_ptrace}" ]
	then
		log_warning "==== Not running UCS library loading test ===="
		return
	fi

	build_ucx_profiling_test

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

	# Test global config list integrity after loading/unloading of UCT
	echo "==== Running test_dlopen_cfg_print ===="
	./test/apps/test_dlopen_cfg_print
}

test_ucp_dlopen() {
	# Make sure UCP library, when opened with dlopen(), loads CMA module
	LIB_CMA=`find ${ucx_inst} -name libuct_cma.so.0`
	if [ -n "$LIB_CMA" ]
	then
		echo "==== Running UCP library loading test ===="
		./test/apps/test_ucp_dlopen | grep 'cma'
	else
		echo "==== Not running UCP library loading test ===="
	fi

	# Test module allow-list
	UCX_MODULES=^ib,rdmacm ./src/tools/info/ucx_info -d |& tee ucx_info_noib.log
	if grep -in "component:\s*ib$" ucx_info_noib.log
	then
		echo "IB module was loaded even though it was disabled"
		exit 1
	fi

	# Test module allow-list passed through ucp_config_modify()
	./test/apps/test_ucp_config -c "UCX_MODULES=^ib,rdmacm" |& tee ucx_config_noib.log
	if grep -in "component:\s*ib$" ucx_config_noib.log
	then
		echo "IB module was loaded even though it was disabled"
		exit 1
	fi
}

test_init_mt() {
	echo "==== Running multi-thread init ===="
	# Each thread requires 5MB. Cap threads number by total available shared memory.
	max_threads=$(df /dev/shm | awk '/shm/ {printf "%d", $4 / 5000}')
	num_threads=$(($max_threads < $(nproc) ? $max_threads : $(nproc)))
	$MAKEP
	for ((i=0;i<10;++i))
	do
		OMP_NUM_THREADS=$num_threads $AFFINITY timeout 5m ./test/apps/test_init_mt
	done
}

test_memtrack() {
	echo "==== Running memtrack test ===="
	UCX_MEMTRACK_DEST=stdout ./test/gtest/gtest --gtest_filter=test_memtrack.sanity

	echo "==== Running memtrack limit test ===="
	UCX_MEMTRACK_DEST=stdout UCX_HANDLE_ERRORS=none UCX_MEMTRACK_LIMIT=512MB ./test/apps/test_memtrack_limit |& grep -C 100 'SUCCESS'
	UCX_MEMTRACK_DEST=stdout UCX_HANDLE_ERRORS=none UCX_MEMTRACK_LIMIT=412MB ./test/apps/test_memtrack_limit |& grep -C 100 'reached'
}

test_unused_env_var() {
	# We must create a UCP worker to get the warning about unused variables
	echo "==== Running ucx_info env vars test ===="
	UCX_IB_PORTS=mlx5_0:1 ./src/tools/info/ucx_info -epw -u t | grep "unused" | grep -q -E "UCX_IB_PORTS"
}

test_env_var_aliases() {
	echo "==== Running MLX5 env var aliases test ===="
	if [[ `./src/tools/info/ucx_info -b | grep -P 'HW_TM *1$'` ]]
	then
		vars=( "TM_ENABLE" "TM_LIST_SIZE" "TX_MAX_BB" )
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

run_gtest_watchdog_test() {
	watchdog_timeout=$1
	sleep_time=$2
	expected_runtime=$3
	expected_err_str="Connection timed out - abort testing"

	echo "==== Running watchdog timeout test ===="

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

run_malloc_hook_gtest() {
	# GTEST_SHARD_INDEX/GTEST_TOTAL_SHARDS should NOT be set

	echo "==== Running malloc hooks mallopt() test, $compiler_name compiler ===="
	$AFFINITY $TIMEOUT env \
		UCX_IB_RCACHE=n \
		MALLOC_TRIM_THRESHOLD_=-1 \
		MALLOC_MMAP_THRESHOLD_=-1 \
		GTEST_FILTER=malloc_hook_cplusplus.mallopt \
			make -C test/gtest test

	echo "==== Running malloc hooks mmap_ptrs test with MMAP_THRESHOLD=16384, $compiler_name compiler ===="
	$AFFINITY $TIMEOUT env \
		MALLOC_MMAP_THRESHOLD_=16384 \
		GTEST_FILTER=malloc_hook_cplusplus.mmap_ptrs \
			make -C test/gtest test
}

#
# Run the test suite (gtest)
# Arguments: <compiler-name> [configure-flags]
#
run_gtest() {
	compiler_name=$1

	export GTEST_RANDOM_SEED=0
	export GTEST_SHUFFLE=1
	# Run UCT tests for TCP over fastest device only
	export GTEST_UCT_TCP_FASTEST_DEV=1
	export OMP_NUM_THREADS=4

	GTEST_EXTRA_ARGS=""
	if [ "$JENKINS_TEST_PERF" == 1 ]
	then
		# Check performance with 10 retries and 2 seconds interval
		GTEST_EXTRA_ARGS="$GTEST_EXTRA_ARGS -p 10 -i 2.0"
	fi
	export GTEST_EXTRA_ARGS

	# Run specific tests
	do_distributed_task 1 4 run_malloc_hook_gtest
	do_distributed_task 2 4 run_gtest_watchdog_test 5 60 300
	do_distributed_task 3 4 test_memtrack

	# Distribute the tests among the workers
	export GTEST_SHARD_INDEX=$worker
	export GTEST_TOTAL_SHARDS=$nworkers
	# Report TOP-20 longest test at the end of testing
	export GTEST_REPORT_LONGEST_TESTS=20

	# Run all tests
	echo "==== Running unit tests, $compiler_name compiler ===="
	$AFFINITY $TIMEOUT make -C test/gtest test

	# Run valgrind tests
	if ! [[ $(uname -m) =~ "aarch" ]] && ! [[ $(uname -m) =~ "ppc" ]] && \
	   ! [[ -n "${JENKINS_NO_VALGRIND}" ]]
	then
		echo "==== Running valgrind tests, $compiler_name compiler ===="

		# Load newer valgrind if naative is older than 3.10
		if ! (echo "valgrind-3.10.0"; valgrind --version) | sort -CV
		then
			module load tools/valgrind-3.12.0
		fi

		$AFFINITY $TIMEOUT_VALGRIND make -C test/gtest test_valgrind
		module unload tools/valgrind-3.12.0
	else
		echo "==== Not running valgrind tests with $compiler_name compiler ===="
	fi

	unset GTEST_REPORT_LONGEST_TESTS
	unset GTEST_TOTAL_SHARDS
	unset GTEST_SHARD_INDEX
	unset GTEST_EXTRA_ARGS
	unset OMP_NUM_THREADS
	unset GTEST_UCT_TCP_FASTEST_DEV
	unset GTEST_SHUFFLE
	unset GTEST_RANDOM_SEED
}

run_gtest_armclang() {
	if [[ $(hostname) =~ '-bf1' ]]
	then
		# Skip armclang test on SoC platform
		return 0
	fi

	if module_load arm-compiler/arm-hpc-compiler
	then
		if armclang -v
		then
			# Force using loaded gcc toolchain instead of host gcc, to avoid
			# compatibility issues
			ARMCLANG_CFLAGS=""
			if [ -n ${GCC_DIR} ]; then
				ARMCLANG_CFLAGS+=" --gcc-toolchain=${GCC_DIR}"
			fi

			# Disable go build, since armclang has some old go compiler.
			build devel --enable-gtest "$@" \
				CC=armclang \
				CXX=armclang++ \
				CFLAGS="${ARMCLANG_CFLAGS}" \
				--without-go

			run_gtest "armclang"
		else
			echo "==== Not running with armclang compiler ===="
			log_warning "armclang compiler is unusable"
		fi

		module unload arm-compiler/arm-hpc-compiler
	else
		echo "==== Not running with armclang compiler ===="
	fi
}

#
# Run the test suite (gtest) in release configuration with small subset of tests
#
run_gtest_release() {
	export GTEST_SHARD_INDEX=0
	export GTEST_TOTAL_SHARDS=1
	export GTEST_RANDOM_SEED=0
	export GTEST_SHUFFLE=1
	export OMP_NUM_THREADS=4

	echo "==== Running unit tests (release configuration) ===="
	# Check:
	# - Important object sizes
	# - Unexpected RNDV test, to cover rkey handling in tag offload flow
	#   (see GH #3827 for details)
	env GTEST_FILTER=\*test_obj_size\*:\*test_ucp_tag_match.rndv_rts_unexp\* \
		$AFFINITY $TIMEOUT make -C test/gtest test

	unset OMP_NUM_THREADS
	unset GTEST_SHARD_INDEX
	unset GTEST_TOTAL_SHARDS
	unset GTEST_RANDOM_SEED
	unset GTEST_SHUFFLE
}

run_ucx_info() {
	echo "==== Running ucx_info ===="

	./src/tools/info/ucx_info -s -f -c -v -y -d -b -p -w -e -uart -m 20M -T -M
}

run_ucx_tl_check() {
	# Test transport selection
	../test/apps/test_ucx_tls.py -p $ucx_inst

	# Test setting many lanes
	UCX_IB_NUM_PATHS=8 \
		UCX_MAX_EAGER_LANES=4 \
		UCX_MAX_RNDV_LANES=4 \
		./src/tools/info/ucx_info -u t -e
}

#
# Run release mode tests
#
run_release_mode_tests() {
	build release --enable-gtest
	test_profiling
	test_ucs_load
	test_ucp_dlopen
	run_gtest_release
}

#
# Run all tests
#
run_tests() {
	export UCX_HANDLE_ERRORS=freeze,bt
	export UCX_ERROR_SIGNALS=SIGILL,SIGSEGV,SIGBUS,SIGFPE,SIGPIPE,SIGABRT
	export UCX_ERROR_MAIL_TO=$ghprbActualCommitAuthorEmail
	export UCX_ERROR_MAIL_FOOTER=$JOB_URL/$BUILD_NUMBER/console
	export UCX_TCP_PORT_RANGE="$((33000 + EXECUTOR_NUMBER * 1000))-$((33999 + EXECUTOR_NUMBER * 1000))"
	export UCX_TCP_CM_REUSEADDR=y

	# Don't cross-connect RoCE devices
	export UCX_IB_ROCE_LOCAL_SUBNET=y
	export UCX_IB_ROCE_SUBNET_PREFIX_LEN=inf

	# load cuda env only if GPU available for remaining tests
	try_load_cuda_env

	# all are running mpi tests
	run_mpi_tests

	# build for devel tests and gtest
	build devel --enable-gtest

	# devel mode tests
	do_distributed_task 0 4 test_unused_env_var
	do_distributed_task 1 4 run_ucx_info
	do_distributed_task 2 4 run_ucx_tl_check
	do_distributed_task 3 4 test_ucs_dlopen
	do_distributed_task 0 4 test_env_var_aliases
	do_distributed_task 1 4 test_malloc_hook
	do_distributed_task 2 4 test_init_mt
	do_distributed_task 3 4 run_ucp_client_server

	# long devel tests
	do_distributed_task 0 4 run_ucp_hello
	do_distributed_task 1 4 run_uct_hello
	do_distributed_task 2 4 run_ucx_perftest
	do_distributed_task 3 4 run_io_demo

	# all are running gtest
	run_gtest "default"

	# build and run gtest with armclang
	run_gtest_armclang

	# release mode tests
	do_distributed_task 0 4 run_release_mode_tests
}

prepare
try_load_cuda_env
if [ -n "$JENKINS_RUN_TESTS" ] || [ -n "$RUN_TESTS" ]
then
	check_machine
	run_tests
fi
