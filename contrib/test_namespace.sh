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

source $(dirname $0)/../buildlib/tools/common.sh

ucx_inst=${WORKSPACE}/install

echo "==== Running namespace tests on $(hostname) ===="

test_namespace() {
	# Make sure to try to use CMA when possible
	# Expect fallback on SYSV
	perftest="$ucx_inst/bin/ucx_perftest -t ucp_get -s 9999999 -n 5"

	echo "==== Running perftest namespace positive tests ===="

	for tls in posix cma,sysv
	do
		echo "==== Running perftest same non-default USER namespace test for $tls ===="

		cmd="UCX_TLS=$tls $perftest -p $server_port"
		step_server_port
		unshare --user bash -c "{ $cmd & sleep 3; $cmd localhost; }"
	done

	for tl in posix cma
	do
		echo "==== Running perftest different PID namespace test for $tl ===="

		cmd="$perftest -p $server_port"
		step_server_port
		sudo unshare --pid --fork sudo -u $USER UCX_TLS=$tl,sysv $cmd &
		sleep 3
		sudo unshare --pid --fork sudo -u $USER UCX_TLS=$tl,sysv $cmd localhost

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
build release
test_namespace
