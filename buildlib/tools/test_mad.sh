#!/bin/bash
set -exE -o pipefail

IMAGE="rdmz-harbor.rdmz.labs.mlnx/ucx/x86_64/rhel8.2/builder:mofed-5.0-1.0.0.0"

if [ -z "$BUILD_SOURCESDIRECTORY" ]; then
    echo "Not running in Azure"
    exit 1
fi
cd "$BUILD_SOURCESDIRECTORY"

build_ucx() {
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
        --user "$(id -u)":"$(id -g)" \
        --name ucx_build_"$BUILD_BUILDID" \
        -e BUILD_SOURCESDIRECTORY="$BUILD_SOURCESDIRECTORY" \
        -v "$PWD":"$PWD" -w "$PWD" \
        -v /hpc/local:/hpc/local \
        $IMAGE \
        bash -c "source ./buildlib/tools/test_mad.sh && build_ucx"
}

docker_run_srv() {
    local test_name="$1"
    HCA=$(detect_active_ib_hca)
    sudo chmod 777 /dev/infiniband/umad*
    docker run \
        --rm \
        --detach \
        --net=host \
        --user "$(id -u)":"$(id -g)" \
        --name ucx_perftest_"$BUILD_BUILDID" \
        -e BUILD_SOURCESDIRECTORY="$BUILD_SOURCESDIRECTORY" \
        -v "$PWD":"$PWD" -w "$PWD" \
        -v /hpc/local:/hpc/local \
        --ulimit memlock=-1:-1 --device=/dev/infiniband/ \
        $IMAGE \
        bash -c "${PWD}/install/bin/ucx_perftest -K ${HCA} > \
            ${PWD}/ucx_perf_srv_${test_name}_${BUILD_BUILDID}.log 2>&1"
}

docker_stop_srv() {
    local test_name="$1"
    cat "${PWD}/ucx_perf_srv_${test_name}_${BUILD_BUILDID}.log"
    docker stop ucx_perftest_"$BUILD_BUILDID" || true
}

set_vars() {
    set +x
    HCA=$(detect_active_ib_hca)
    # Replace ':' with space for 'ibstat' format
    HCA_DEV=${HCA/:/ }
    # shellcheck disable=SC2086
    LID=$(ibstat $HCA_DEV | grep Base | awk '{print $NF}')
    # shellcheck disable=SC2086
    GUID=$(ibstat $HCA_DEV | grep GUID | awk '{print $NF}')
    echo "##vso[task.setvariable variable=LID;isOutput=true]$LID"
    echo "##vso[task.setvariable variable=GUID;isOutput=true]$GUID"
    echo "##vso[task.setvariable variable=HCA;isOutput=true]$HCA"
    echo "LID: $LID"
    echo "GUID: $GUID"
    echo "HCA: $HCA"
}

run_mad_test() {
    local ib_address="$1"
    sudo chmod 777 /dev/infiniband/umad*
    "$PWD"/install/bin/ucx_perftest -t tag_bw -e -K "$HCA" -e "$ib_address"
}

detect_active_ib_hca() {
  ibv_devinfo | awk '
    /hca_id:/ {hca=$2}
    /port:/ {port=$2}
    /state:/ {state=$2}
    /link_layer:/ {link_layer=$2}
    state == "PORT_ACTIVE" && link_layer == "InfiniBand" {
      print hca ":" port
      found=1
      exit
    }
    END {
      if (!found) {
        echo "Error: No active InfiniBand interface found"
        exit 1
      }
    }
  '
}
