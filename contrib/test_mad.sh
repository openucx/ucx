#!/bin/bash
set -exE -o pipefail

export HCA="mlx5_0:1"
cd "$BUILD_SOURCESDIRECTORY"

run_mad_server() {
    srv_stop
    funcname
    sudo -E bash -c 'envsubst < "contrib/ucx_perftest.service" \
        > /etc/systemd/system/ucx_perftest.service'
    sudo systemctl daemon-reload
    srv_start
}

build_ucx() {
    funcname
    ./autogen.sh
    ./contrib/configure-release \
        --prefix="$PWD"/install \
        --with-mad \
        --without-valgrind \
        --without-go \
        --without-java
    make -s -j"$(nproc)"
    make install
}

setup() {
    funcname
    sudo chmod 777 /dev/infiniband/umad*
}

set_vars() {
    set +x
    HCA=${HCA/:/ } # Replace ':' with space
    # shellcheck disable=SC2086
    LID=$(ibstat $HCA | grep Base | awk '{print $NF}')
    # shellcheck disable=SC2086
    GUID=$(ibstat $HCA | grep GUID | awk '{print $NF}')
    echo "LID: $LID"
    echo "GUID: $GUID"
    echo "##vso[task.setvariable variable=LID;isOutput=true]$LID"
    echo "##vso[task.setvariable variable=GUID;isOutput=true]$GUID"
}

run_mad_test_lid() {
    funcname
    "$PWD"/install/bin/ucx_perftest -t tag_bw -e -K "$HCA" -e lid:"$LID"
}

run_mad_test_guid() {
    funcname
    "$PWD"/install/bin/ucx_perftest -t tag_bw -e -K "$HCA" guid:"$GUID"
}

srv_start() {
    funcname
    sudo systemctl start ucx_perftest
    sudo systemctl status ucx_perftest
}

srv_stop() {
    funcname
    set +e
    sudo systemctl status ucx_perftest
    sudo systemctl stop ucx_perftest
    set -e
}

funcname() {
    set +x
    echo "==== Running: ${FUNCNAME[1]} ===="
    set -x
}

"$@"
