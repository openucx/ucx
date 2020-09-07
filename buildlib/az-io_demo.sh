#!/bin/bash -leE

# avoid Azure error: TERM environment variable not set
export TERM=xterm

basedir=$(cd $(dirname $0) && pwd)
workspace=${WORKSPACE:="$basedir"}
cd "$workspace"

echo "Running $0 $*..."
eval "$*"
source "${workspace}/az-helpers.sh"

server_ip=${server_ip:=""}
duration=${duration:=2}
iface=${iface:="bond0"}

export UCX_MAX_EAGER_LANES=2
export UCX_TLS=rc
export UCX_IB_SEG_SIZE=2k
export UCX_IB_RX_QUEUE_LEN=1024
export UCX_RC_MAX_RD_ATOMIC=16
export UCX_RC_ROCE_PATH_FACTOR=2
export UCX_SOCKADDR_CM_ENABLE=y
export UCX_RC_MAX_GET_ZCOPY=32k
export UCX_RC_TX_NUM_GET_BYTES=256K

## run server
if [ "x$server_ip" = "x" ]; then
    ip addr show ${iface}
    server_ip=$(get_ip ${iface})
    azure_set_variable "server_ip" "$server_ip"
    echo "Starting server on IP ${server_ip}"

    server_cmd="${workspace}/../test/apps/iodemo/io_demo"
    if ! "${server_cmd}" |& add_timestamp &>server.log & then
        error "Failed to run server command ${server_cmd}"
    fi

    # wait for io_demo to start
    sleep 5

    server_pid=$(pgrep -u "$USER" -f 'apps/iodemo')
    num_pids=$(echo "${server_pid}" | wc -w)
    if [ ${num_pids} -ne 1 ]; then
        ps -f -U "$USER"  # show all runing processes
        error "Expected 1 running server, found ${num_pids}"
    fi

    echo "Server is running, PID='$server_pid'"
    azure_set_variable "server_pid" "$server_pid"

    # double check the process is running
    sleep 5
    if ! kill -0 "$server_pid"; then
        cat server.log
        error "Failed to start server"
    fi

    exit 0
fi

## run client

timeout="$(( duration - 1 ))m"

echo "Client connecting to server at IP $server_ip"
echo "Timeout is $timeout"

if ! "${workspace}/../test/apps/iodemo/io_demo" -l $timeout -i 10000000 "$server_ip"; then
    error "Failed to start client"
fi
