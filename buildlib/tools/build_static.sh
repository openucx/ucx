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
# Build with static library
#
build_static() {
	az_module_load dev/libnl
	az_module_load dev/numactl

	${WORKSPACE}/contrib/configure-devel --prefix=$ucx_inst
	$MAKEP
	$MAKEP install

	# Build test applications
	SAVE_PKG_CONFIG_PATH=$PKG_CONFIG_PATH
	export PKG_CONFIG_PATH=$ucx_inst/lib/pkgconfig:$PKG_CONFIG_PATH

	$MAKE -C test/apps/uct_info

	export PKG_CONFIG_PATH=$SAVE_PKG_CONFIG_PATH

	# Run test applications and check script
	SAVE_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
	export LD_LIBRARY_PATH=$ucx_inst/lib:$LD_LIBRARY_PATH

	cd ./test/apps/uct_info

	./uct_info
	./uct_info_static

	${WORKSPACE}/buildlib/tools/check_tls.sh $EXTRA_TLS

	# Set port number for hello_world applications
	server_port=$((10000 + (1000 * EXECUTOR_NUMBER)))
	server_port_arg="-p $server_port"

	for tls in tcp $RUN_TLS; do
		echo UCX_TLS=$tls
		UCX_TLS=$tls ./ucp_hello_world_static ${server_port_arg} &
		PID=$!
		# allow server to start
		sleep 10
		UCX_TLS=$tls ./ucp_hello_world_static ${server_port_arg} -n localhost
		wait ${pid} || true
	done

	export LD_LIBRARY_PATH=$SAVE_LD_LIBRARY_PATH

	az_module_unload dev/numactl
	az_module_unload dev/libnl
}

az_init_modules
prepare_build

# Don't cross-connect RoCE devices
export UCX_IB_ROCE_SUBNET_PREFIX_LEN=inf
build_static

