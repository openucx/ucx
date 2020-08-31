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
    echo "Server's IP is ${server_ip}"

    if ! "${workspace}/../test/apps/iodemo/io_demo" |& add_timestamp &>server.log & then
        error "Failed to start server"
    fi
    sleep 1     # wait for .libs/li-io_demo to start
    pgrep -u "$USER" -fa 'apps/iodemo'
    server_pid=$(pgrep -u "$USER" -f 'apps/iodemo')
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

echo "Server IP is $server_ip"
echo "Timeout is $timeout"

if ! "${workspace}/../test/apps/iodemo/io_demo" -l $timeout -i 10000000 "$server_ip"; then
    error "Failed to start client"
fi
