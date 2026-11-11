#!/bin/bash
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
set -eo pipefail

if [ "$#" -eq 0 ]; then
    result=0
    for pr in 12311 12312; do
        bash "$0" "$pr" || result=1
    done
    exit "$result"
fi

verification_root=$PWD
pr=$1
case $pr in
    12311) revision=b3e859d0f7741c3cde87c7d838744c9bc95d529c
           tests=(mempool loopback) ;;
    12312) revision=c5af3fef50b27127225097593be82eb810e52f14
           tests=(cpu_mask) ;;
    *) exit 2 ;;
esac
source ./buildlib/az-helpers.sh
az_init_modules
RUNNING_IN_AZURE=no
try_load_cuda_env
test "$have_cuda" != no
set -u
nvidia-smi -L
compgen -G '/dev/infiniband/uverbs*' >/dev/null
python3 -c 'import os; assert len(os.sched_getaffinity(0)) >= 64, "Need 64 affinity CPUs"'
ulimit -l unlimited
unset UCX_TLS UCX_NET_DEVICES UCX_CUDA_IPC_ENABLE_SAME_PROCESS
unset GTEST_SHARD_INDEX GTEST_TOTAL_SHARDS
export UCX_HANDLE_ERRORS=bt OMP_NUM_THREADS=4 GTEST_MAX_IB_SL=4
source_dir=$verification_root/regression-gpu-$pr
git clone --no-checkout https://github.com/openucx/ucx.git "$source_dir"
git -C "$source_dir" fetch --no-tags origin "$revision"
git -C "$source_dir" checkout --detach "$revision"
test "$(git -C "$source_dir" rev-parse HEAD)" = "$revision"
cd "$source_dir"
./autogen.sh
mkdir build-focused
cd build-focused
../contrib/configure-devel --prefix="$PWD/install" --with-cuda="$have_cuda" \
    --with-verbs --without-java --without-valgrind
make -j16
result=0
for test_name in "${tests[@]}"; do
    case $test_name in
        mempool) filter='cuda_ipc/test_cuda_ipc_md.mkey_pack_mempool/*' ;;
        loopback) filter='cuda_ipc/test_uct_loopback_cuda.envelope/*' ;;
        cpu_mask) filter='all/test_worker_cpu_mask.all_cpus/*' ;;
    esac
    printf 'VERIFY PR=%s SHA=%s TEST=%s HOST=%s\n' "$pr" "$revision" "$filter" "$(hostname)"
    status=0
    output=$(UCX_LOG_LEVEL=diag timeout 1800 make -C test/gtest test \
        GTEST_FILTER="$filter" \
        GTEST_EXTRA_ARGS="-p 10 -i 2.0 --gtest_output=xml:$PWD/$test_name.xml" 2>&1) || status=$?
    printf '%s\n' "$output"
    if ! python3 - "$test_name" "$test_name.xml" "$status" 3<<<"$output" <<'PY'
import os
import re
import sys
import xml.etree.ElementTree as ET

name, report, status = sys.argv[1:]
text = os.fdopen(3).read()
assert status == '0', 'NOT VERIFIED: test command failed'
cases = list(ET.parse(report).getroot().iter('testcase'))
assert cases, 'NOT VERIFIED: zero selected tests'
assert not any(c.find(t) is not None for c in cases for t in ('failure', 'error', 'skipped'))
assert not re.search(r'\[\s*SKIP(?:PED)?\s*\]', text), 'NOT VERIFIED: UCX skipped the test'
if name == 'mempool':
    assert not os.path.exists('/dev/nvidia-caps-imex-channels/channel0'), 'NOT VERIFIED: IMEX present'
    assert 'cuMemPoolExportToShareableHandle' in text and 'operation not permitted' in text, \
        'NOT VERIFIED: missing failed optional FABRIC export'
elif name == 'loopback':
    assert re.search(r'(put|get) zcopy bw\s*:\s*[0-9.]+ MB/sec', text), \
        'NOT VERIFIED: no completed CUDA IPC transfer'
else:
    visited = {int(cpu) for cpu in re.findall(r'create entity on cpu (\d+)', text)}
    assert visited == os.sched_getaffinity(0), 'NOT VERIFIED: incomplete CPU coverage'
print('VERIFIED TEST=%s tests=%s failures=0 skipped=0' % (name, len(cases)))
PY
    then
        status=1
    fi
    [ "$status" -eq 0 ] || result=1
done
exit "$result"
