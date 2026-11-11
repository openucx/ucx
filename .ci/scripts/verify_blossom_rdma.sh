#!/bin/bash
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
set -euo pipefail

case "${1:-}" in
    pci-vf)
        revision=b3e859d0f7741c3cde87c7d838744c9bc95d529c
        link_layer=InfiniBand
        filter='*test_pci_bw.get_pci_bw*:test_topo_pci_bw.virtual_function'
        configure_args=(--enable-asan)
        ;;
    roce-ipv6)
        revision=c5af3fef50b27127225097593be82eb810e52f14
        link_layer=Ethernet
        filter='rdmacm/test_uct_sockaddr_err_handle_non_exist_ip.conn_to_non_exist_ip*'
        configure_args=()
        ;;
    *) echo 'Expected pci-vf or roce-ipv6' >&2; exit 2 ;;
esac

vf_device=
for candidate in /sys/class/infiniband/*; do
    [[ -e "$candidate/device/physfn" ]] || continue
    device_info=$(ibv_devinfo -d "${candidate##*/}" -i 1 2>/dev/null) || continue
    grep -q 'PORT_ACTIVE' <<< "$device_info" || continue
    grep -Eq "link_layer:[[:space:]]+$link_layer" <<< "$device_info" || continue
    vf_device=${candidate##*/}
    printf '%s\n' "$device_info"
    echo "VF $vf_device: $(readlink -f "$candidate/device")"
    echo "PF: $(readlink -f "$candidate/device/physfn")"
    break
done
[[ -n "$vf_device" ]] || { echo 'UNAVAILABLE: no usable requested VF'; exit 1; }
export UCX_NET_DEVICES="$vf_device:1"
export REGRESSION_NETDEV= REGRESSION_IPV6=
if [[ "$1" == roce-ipv6 ]]; then
    for net_path in /sys/class/net/*; do
        [[ $(readlink -f "$net_path/device") == $(readlink -f "$candidate/device") ]] || continue
        addresses=$(ip -6 -o addr show dev "${net_path##*/}" scope global)
        [[ -n "$addresses" ]] || continue
        printf '%s\n' "$addresses"
        export REGRESSION_NETDEV=${net_path##*/}
        export REGRESSION_IPV6=$(awk 'NR == 1 {split($4, addr, "/"); print addr[1]}' <<< "$addresses")
        break
    done
    [[ -n "$REGRESSION_IPV6" ]] || { echo 'UNAVAILABLE: VF has no global IPv6'; exit 1; }
fi

source_dir=$(mktemp -d "$PWD/rdma-$1.XXXXXX")
git -C "$source_dir" init -q
git -C "$source_dir" fetch --depth=1 https://github.com/openucx/ucx.git "$revision"
git -C "$source_dir" checkout --detach FETCH_HEAD
[[ $(git -C "$source_dir" rev-parse HEAD) == "$revision" ]]
echo "REGRESSION $1 EXACT_HEAD=$revision"
cd "$source_dir"
./autogen.sh
mkdir build-devel
cd build-devel
../contrib/configure-devel --without-java --without-go --without-cuda \
    --without-rocm --without-valgrind --with-verbs --with-rdmacm \
    --disable-examples --disable-test-apps "${configure_args[@]}"
make -j8
export UCX_HANDLE_ERRORS=bt GTEST_SHARD_INDEX=0 GTEST_TOTAL_SHARDS=1
export GTEST_FILTER="$filter"
python3 - "$1" <<'PY'
import os
import re
import subprocess
import sys

kind = sys.argv[1]
current = None
passed = set()
skipped = False
ipv6_test = None
process = subprocess.Popen(
    ["make", "-C", "test/gtest", "test", "GTEST_EXTRA_ARGS=--gtest_color=no"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in process.stdout:
    print(line, end="", flush=True)
    started = re.match(r"\[ RUN +\] (\S+)", line)
    if started:
        current = started.group(1)
    if re.search(r"\[\s*SKIP(?:PED)?\s*\]", line):
        skipped = True
    if (kind == "roce-ipv6" and current and
            " interface " + os.environ["REGRESSION_NETDEV"] in line and
            "RDMA_CM_SOURCE_ADDRESS=" + os.environ["REGRESSION_IPV6"] in line):
        ipv6_test = current
    finished = re.match(r"\[ +OK +\] (\S+)", line)
    if finished:
        passed.add(finished.group(1))
status = process.wait()
valid = not status and not skipped and bool(passed)
if kind == "pci-vf":
    valid = valid and any("test_pci_bw.get_pci_bw" in t for t in passed)
    valid = valid and "test_topo_pci_bw.virtual_function" in passed
else:
    valid = valid and ipv6_test in passed
print(f"REGRESSION_RESULT case={kind} passed={len(passed)} skipped={skipped} valid={valid}")
sys.exit(0 if valid else 1)
PY
