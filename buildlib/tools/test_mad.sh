#!/bin/bash
set -exE -o pipefail

IMAGE="rdmz-harbor.rdmz.labs.mlnx/ucx/x86_64/rhel8.2/builder:mofed-5.0-1.0.0.0"
cd "$BUILD_SOURCESDIRECTORY"

node_setup() {
    funcname
    sudo chmod 777 /dev/infiniband/umad*
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

build_ucx_in_docker() {
    docker run --rm \
        --name ucx_build_"$BUILD_BUILDID" \
        -v "$PWD":"$PWD" -w "$PWD" \
        $IMAGE \
        bash -c "source ./buildlib/tools/test_mad.sh && build_ucx"

    sudo chown -R swx-azure-svc:ecryptfs "$PWD"
}

docker_run_srv() {
    detect_hca
    docker run --rm \
        --detach \
        --net=host \
        -e HCA="$HCA" \
        --name ucx_perftest_"$BUILD_BUILDID" \
        -v "$PWD":"$PWD" -w "$PWD" \
        --gpus all --ulimit memlock=-1:-1 --device=/dev/infiniband/ \
        $IMAGE \
        bash -c "source ./buildlib/tools/test_mad.sh && \
                ${PWD}/install/bin/ucx_perftest -K ${HCA}"
}

set_vars() {
    set +x
    detect_hca
    # Replace ':' with space for 'ibstat' format
    HCA=${HCA/:/ }
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
    detect_hca
    "$PWD"/install/bin/ucx_perftest -t tag_bw -e -K "$HCA" -e lid:"$LID"
}

run_mad_test_guid() {
    funcname
    detect_hca
    "$PWD"/install/bin/ucx_perftest -t tag_bw -e -K "$HCA" guid:"$GUID"
}

funcname() {
    set +x
    echo "==== Running: ${FUNCNAME[1]} ===="
    set -x
}

detect_hca() {
    echo "Detect first active HCA port"
    HCA="$(ibv_devinfo | awk '/hca_id:/ {hca=$2} /port:/ {port=$2} /PORT_ACTIVE/ {print hca ":" port; exit}')"
    export HCA
}

"$@"
