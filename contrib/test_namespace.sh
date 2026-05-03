#!/bin/bash -eEx
#
# Testing script for UCX namespace related functionality
#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2023. ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#
#
# Environment variables set by Azure CI:
#  - WORKSPACE           : path to work dir
#

source $(dirname $0)/../buildlib/az-helpers.sh
source $(dirname $0)/../buildlib/tools/common.sh

ucx_inst=${WORKSPACE}/install

echo "==== Running namespace tests on $(hostname) ===="

test_namespace_pid() {
	local tl=$1
	local mem_type=$2
	local test_type=$3
	local base_perftest=$4
	local expected_tl=$5
	local config
	local base_cmd
	local unshare_cmd

	echo "==== Running perftest different PID namespace test for $tl ===="

	base_cmd="$base_perftest -t $test_type -m $mem_type -p $server_port"
	config="UCX_PROTO_INFO=y UCX_TLS=$tl,sysv"

    # TODO: remove this once we have a way to test with multiple GPUs
    config="$config CUDA_VISIBLE_DEVICES=0"
	
	unshare_cmd="sudo unshare --mount-proc --pid --fork sudo -u $USER $config $base_cmd"
	
	step_server_port
	$unshare_cmd &
	sleep 3
	output=$($unshare_cmd localhost)

	if [ "X$expected_tl" != "X" ]
	then
		echo "$output" | grep -q "$expected_tl"
	fi
}

test_namespace() {
	# Make sure to try to use CMA when possible
	# Expect fallback on SYSV
	local base_perftest="$ucx_inst/bin/ucx_perftest -s 9999999 -n 5"
	local perftest="$base_perftest -t ucp_get"
	local cmd

	echo "==== Running perftest namespace positive tests ===="

	if [ "X$have_cuda" != "Xno" ] 
	then
		test_namespace_pid cuda_ipc,cuda_copy cuda ucp_put_bw "$base_perftest" cuda_ipc
	    # TODO: remove this once CUDA driver hang on GPU CI is fixed
		return 0
	fi

	for tls in posix cma,sysv
	do
		echo "==== Running perftest same non-default USER namespace test for $tls ===="

		cmd="UCX_TLS=$tls $perftest -p $server_port"
		step_server_port
		unshare --user bash -c "{ $cmd & sleep 3; $cmd localhost; }"
	done

	test_namespace_pid posix host ucp_get "$base_perftest"
	test_namespace_pid cma host ucp_get "$base_perftest"

	for tl in posix cma
	do
		echo "==== Running perftest different USER namespace test for $tl ===="
		cmd="$perftest -p $server_port"
		step_server_port
		UCX_TLS=$tl,sysv unshare --user $cmd &
		sleep 3
		UCX_TLS=$tl,sysv unshare --user $cmd localhost
	done

	echo "==== Running perftest different USER namespace test for posix non proc link ===="
	cmd="$perftest -p $server_port"
	step_server_port
	UCX_TLS="posix" UCX_POSIX_USE_PROC_LINK=n unshare --user $cmd &
	sleep 3
	UCX_TLS="posix" UCX_POSIX_USE_PROC_LINK=n unshare --user $cmd localhost
}

prepare
try_load_cuda_env
build release
test_namespace
