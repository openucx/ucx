#!/bin/bash -eExl

WORKSPACE=${WORKSPACE:=$PWD}
# build in local directory which goes away when docker exits
ucx_build_dir=$HOME/${BUILD_ID}/build
ucx_inst=$ucx_build_dir/install
CUDA_MODULE="dev/cuda12.2.2"
GDRCOPY_MODULE="dev/gdrcopy2.3.1-1_cuda12.2.2"
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
parallel_jobs=${parallel_jobs:-4}
[ $parallel_jobs -gt $num_cpus ] && parallel_jobs=$num_cpus
num_pinned_threads=$(nproc)
[ $parallel_jobs -gt $num_pinned_threads ] && parallel_jobs=$num_pinned_threads

MAKE="make V=1"
MAKEP="make V=1 -j${parallel_jobs}"
export AUTOMAKE_JOBS=$parallel_jobs

#
# Set initial port number for client/server applications to be updated with
# function below
#
server_port_range=1000
server_port_min=$((10500 + EXECUTOR_NUMBER * server_port_range))
server_port_max=$((server_port_min + server_port_range))
server_port=${server_port_min}

step_server_port() {
	# Cycle server_port between (server_port_min)..(server_port_max-1)
	server_port=$((server_port + 1))
	server_port=$((server_port >= server_port_max ? server_port_min : server_port))
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
		config_args+=" --with-iodemo-cuda"
	fi

	../contrib/configure-${mode} ${config_args} "$@"
	make_clean
	$MAKEP
	$MAKEP install
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
# Get IB devices on state Active
#
get_active_ib_devices() {
	get_ib_devices PORT_ACTIVE
}

#
# Check host state
#
check_machine() {
	# Check IB devices on state INIT
	init_dev=$(get_ib_devices PORT_INIT)
	if [ -n "${init_dev}" ]
	then
		echo "${init_dev} have state PORT_INIT"
		exit 1
	fi

	# Log machine status
	lscpu
	uname -a
	free -m
	ofed_info -s || true
	ibv_devinfo -v || true
	show_gids || true
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
