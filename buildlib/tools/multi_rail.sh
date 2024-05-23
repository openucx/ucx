#!/bin/bash
set -exE -o pipefail

IMAGE="rdmz-harbor.rdmz.labs.mlnx/ucx/x86_64/rhel8.2/builder:mofed-5.0-1.0.0.0"

if [ -z "$BUILD_SOURCESDIRECTORY" ]; then
    echo "Not running in Azure"
    exit 1
fi
cd "$BUILD_SOURCESDIRECTORY"

build_ucx() {
    gcc --version
    ./autogen.sh
    ./contrib/configure-release \
        --prefix="$PWD"/install \
        --without-valgrind \
        --without-go \
        --without-java
    make -s -j "$(nproc)"
    make install
}

build_ucx_in_docker() {
    docker run --rm \
        --user "$(id -u)":"$(id -g)" \
        --name ucx_build_"$BUILD_BUILDID" \
        -e BUILD_SOURCESDIRECTORY="$BUILD_SOURCESDIRECTORY" \
        -v "$PWD":"$PWD" -w "$PWD" \
        -v /hpc/local:/hpc/local \
        $IMAGE \
        bash -c "source ./buildlib/tools/multi_rail.sh && build_ucx"
}

docker_run_srv() {
    local test_name="$1"
    "${PWD}"/install/bin/ucx_info -c | grep TCP_EP_BIND

    docker run \
        --rm \
        --detach \
        --net=host \
        --user "$(id -u)":"$(id -g)" \
        --name multi_rail_srv_"$BUILD_BUILDID" \
        -e BUILD_SOURCESDIRECTORY="$BUILD_SOURCESDIRECTORY" \
        -v "$PWD":"$PWD" \
        -w "$PWD" \
        -v /hpc/local:/hpc/local \
        $IMAGE \
        bash -c "UCX_TCP_EP_BIND_SRC_ADDR=$UCX_TCP_EP_BIND_SRC_ADDR \
                UCX_TLS=tcp \
                UCX_NET_DEVICES=ib0,ib1 \
                ${PWD}/install/bin/ucx_perftest -t ucp_am_bw > \
                ${PWD}/multi_rail_srv_${test_name}_${BUILD_BUILDID}.log 2>&1"
}

set_vars() {
    set +x
    # Find IP of the first available HCA port
    NIC=$(ibdev2netdev | awk '/Up/ {print $5}' | head -n 1)
    SRV_IP=$(ip addr show "$NIC" | awk '/inet / {print $2}' | cut -d "/" -f 1)
    echo "##vso[task.setvariable variable=SRV_IP;isOutput=true]$SRV_IP"
    echo "SRV_IP: $SRV_IP"
}

run_client() {
    local SRV_IP="$1"
    "${PWD}"/install/bin/ucx_info -c | grep TCP_EP_BIND

    UCX_TLS=tcp \
        UCX_NET_DEVICES=ib0,ib1 \
        UCX_LOG_LEVEL=debug \
        "${PWD}"/install/bin/ucx_perftest "$SRV_IP" \
        -t ucp_am_bw \
        -s 3276800 \
        -n 10 |
        grep tcp_ep |
        grep accepted |
        tee multi-rail_"$BUILD_BUILDID".log
}

assert() {
    cat multi-rail_"$BUILD_BUILDID".log ## DEBUG
    local TEST_TYPE="$1"

    if [ "$(wc -l <multi-rail_"$BUILD_BUILDID".log)" -lt 2 ]; then
        echo "Error: Dual connection was NOT established!"
        exit 1
    else
        IP1=$(awk '/from/ {print $11}' multi-rail_"$BUILD_BUILDID".log | head -n 1 | cut -d ':' -f 1)
        IP2=$(awk '/from/ {print $11}' multi-rail_"$BUILD_BUILDID".log | tail -n 1 | cut -d ':' -f 1)

        if [[ "$TEST_TYPE" == "negative" && "$IP1" != "$IP2" ]]; then
            echo "Error: Expected single-rail but got multi-rail"
            exit 1
        elif [[ "$TEST_TYPE" == "positive" && "$IP1" == "$IP2" ]]; then
            echo "Error: Expected multi-rail but got single-rail"
            exit 1
        fi
    fi

    rm -f multi-rail_"$BUILD_BUILDID".log
}
