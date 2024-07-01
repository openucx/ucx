#!/bin/bash
set -exE -o pipefail

IMAGE="rdmz-harbor.rdmz.labs.mlnx/ucx/x86_64/rhel8.2/builder:mofed-5.0-1.0.0.0"

if [ -z "$BUILD_SOURCESDIRECTORY" ]; then
    echo "Not running in Azure"
    exit 1
fi
cd "$BUILD_SOURCESDIRECTORY"
source "${PWD}"/buildlib/az-helpers.sh

get_nics() {
    # Get NIC names for the first 'n' available interfaces
    local n=$1
    ibdev2netdev | grep '(Up)' | awk '{print $5}' | head -n "$n" | paste -sd ','
}

NICs=$(get_nics 2)

set_vars() {
    set +x
    local NIC1
    NIC1=$(get_nics 1)
    SRV_IP=$(ip addr show "$NIC1" | awk '/inet / {print $2}' | cut -d "/" -f 1)
    echo "##vso[task.setvariable variable=SRV_IP;isOutput=true]$SRV_IP"
    echo "SRV_IP: $SRV_IP"
}

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
    if [ -z "$1" ]; then
        error "Required param is missing"
    fi
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
                UCX_NET_DEVICES=$NICs \
                ${PWD}/install/bin/ucx_perftest -t ucp_am_bw > \
                ${PWD}/multi_rail_srv_${test_name}_${BUILD_BUILDID}.log 2>&1"
}

print_srv_log() {
    local test_name="$1"
    printf "\n======= Server log for test: %s =======\n" "$test_name"
    cat "${PWD}/multi_rail_srv_${test_name}_${BUILD_BUILDID}.log"
}

get_counters() {
    for NIC in ${NICs//,/ }; do
        # ethtool -S "$NIC" | awk '/tx_bytes:/ {print $2}'
        cat /sys/class/net/"$NIC"/statistics/tx_bytes
    done
}

run_client() {
    if [ -z "$1" ] || [ -z "$2" ]; then
        error "Required params are missing"
    fi

    TEST_TYPE="$1"
    local SRV_IP="$2"

    "${PWD}"/install/bin/ucx_info -c | grep TCP_EP_BIND

    mapfile -t counters_before < <(get_counters)

    UCX_TLS=tcp \
        UCX_TCP_EP_BIND_SRC_ADDR=$UCX_TCP_EP_BIND_SRC_ADDR \
        UCX_NET_DEVICES="$NICs" \
        UCX_LOG_LEVEL=debug \
        "${PWD}"/install/bin/ucx_perftest "$SRV_IP" \
        -t ucp_am_bw \
        -s 3276800 \
        -n 100 |
        grep tcp_ep |
        grep accepted |
        tee "${PWD}/multi_rail_client_${TEST_TYPE}_${BUILD_BUILDID}.log"

    mapfile -t counters_after < <(get_counters)
    # assert_ips
    print_counters "${counters_before[*]}" "${counters_after[*]}"
    assert_counters "${counters_before[*]}" "${counters_after[*]}"
}

assert_ips() {
    local LOG_FILE="${PWD}/multi_rail_client_${TEST_TYPE}_${BUILD_BUILDID}.log"

    if [ "$(wc -l <"$LOG_FILE")" -lt 2 ]; then
        error "Dual connection was not established"
    else
        IP1=$(awk '/from/ {print $11}' "$LOG_FILE" | head -n 1 | cut -d ':' -f 1)
        IP2=$(awk '/from/ {print $11}' "$LOG_FILE" | tail -n 1 | cut -d ':' -f 1)

        if [[ "$TEST_TYPE" == "negative" && "$IP1" != "$IP2" ]]; then
            error "Assert IPs failed: Expected single-rail but got multi-rail"
        elif [[ "$TEST_TYPE" == "positive" && "$IP1" == "$IP2" ]]; then
            error "Assert IPs failed: Expected multi-rail but got single-rail"
        fi
    fi
}

print_counters() {
    set +x
    local counters_before=($1)
    local counters_after=($2)
    local nics=(${NICs//,/ })

    printf '%*s\n' 85 '' | tr ' ' '-'
    printf "\n%-10s %-20s %-20s %-20s\n" "NIC" "Before" "After" "Diff"
    for ((i = 0; i < ${#counters_before[@]}; i++)); do
        local nic="${nics[$i]}"
        local before="${counters_before[$i]}"
        local after="${counters_after[$i]}"
        local diff=$((after - before))
        printf "%-10s %-20s %-20s %-20s\n" "$nic" "$before" "$after" "$diff"
    done
}

assert_counters() {
    local counters_before=($1)
    local counters_after=($2)
    local nics=(${NICs//,/ })
    local diffs=()
    local margin_percentage=0.05

    for ((i = 0; i < ${#counters_before[@]}; i++)); do
        local before="${counters_before[$i]}"
        local after="${counters_after[$i]}"
        local diff=$((after - before))
        diffs+=("$diff")
    done

    if [ "${#diffs[@]}" -ge 2 ]; then
        local margin
        local lower_bound
        local upper_bound
        local within_margin
        local diff1="${diffs[0]}"
        local diff2="${diffs[1]}"
        margin=$(bc <<<"scale=2; $diff1 * $margin_percentage")
        lower_bound=$(bc <<<"$diff1 - $margin")
        upper_bound=$(bc <<<"$diff1 + $margin")
        within_margin=$(bc <<<"$diff2 >= $lower_bound && $diff2 <= $upper_bound")

        if [ "$TEST_TYPE" == "negative" ] && [ "$within_margin" -eq 1 ]; then
            error "Assert counters failed: Expected single-rail but got multi-rail"
        elif [ "$TEST_TYPE" == "positive" ] && [ "$within_margin" -eq 0 ]; then
            error "Assert counters failed: Expected multi-rail but got single-rail"
        fi
    fi
}
