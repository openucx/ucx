#!/bin/bash -eEx
#
# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See file LICENSE for terms.
#

exe_name=$1
client_opt=$2
common_opt=$3
port=$((10000 + 1000 * ${AZP_AGENT_ID} + 100 * ${WIRE_COMPAT_STAGE_ID}))

export UCX_CM_REUSEADDR=y UCX_LOG_LEVEL=info UCX_WARN_UNUSED_ENV_VARS=n
export UCX_IB_ROCE_LOCAL_SUBNET=y

exe_cmd="stdbuf -oL ${exe_name} -p ${port}"

# Test server is legacy, client is master
LD_LIBRARY_PATH=${UCX_LEGACY_LIB_PATH} ${exe_cmd} ${common_opt} &
server_pid=$!
sleep 5
LD_LIBRARY_PATH=${UCX_PR_LIB_PATH} ${exe_cmd} ${common_opt} ${client_opt} 127.0.0.1
if ! kill -9 ${server_pid}; then echo "server already terminated"; fi

exe_cmd="stdbuf -oL ${exe_name} -p $((port + 10))"

# Test server is master, client is legacy
LD_LIBRARY_PATH=${UCX_PR_LIB_PATH} ${exe_cmd} ${common_opt} &
server_pid=$!
sleep 5
LD_LIBRARY_PATH=${UCX_LEGACY_LIB_PATH} ${exe_cmd} ${common_opt} ${client_opt} 127.0.0.1
if ! kill -9 ${server_pid}; then echo "server already terminated"; fi
