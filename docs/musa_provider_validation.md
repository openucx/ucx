# UCX MUSA provider validation

This document is the validation runbook for the optional `musa_cpy` memory
domain and `musa_copy` transport. It is written for MUSA SDK 5.2.0 from the
MThreads distribution and for one fixed UCX source revision. The provider
must not be described as a network MVP until the hardware-gated checks below
are complete.

## Fixed build identity

Record all values in the validation log:

```sh
git rev-parse HEAD
git describe --always --dirty
sha256sum /path/to/ucx-source.tar.gz
uname -a
cat /etc/os-release
cc --version
```

The MUSA build must use one SDK root for headers, `libmusart`, `libmusa`, and
the runtime loader:

```sh
./autogen.sh
./configure --prefix=/opt/ucx-musa \
    --with-musa=/opt/musa/5.2.0 \
    --enable-mt
make -j
make install
```

Verify that headers, UCX modules, and libraries resolve to the same build:

```sh
readelf -d /opt/ucx-musa/lib/libuct_musa.so
ldd /opt/ucx-musa/lib/libuct_musa.so
ldd /opt/ucx-musa/lib/libuct_ib.so
pkg-config --modversion musa 2>/dev/null || true
```

The expected MUSA provider contract is:

| Item | Expected value |
| --- | --- |
| UCX memory type | `musa` / `UCS_MEMORY_TYPE_MUSA` |
| MUSA runtime | 5.2.0 |
| local module | `musa_cpy` |
| local transport | `musa_copy` |
| direct path | `mt-peermem` or IB DMA-BUF |
| fallback | provider-internal host staging only when explicitly declared |

## Capability and topology evidence

Collect the runtime and UCX capabilities from the same environment used by
the test:

```sh
ucx_info -v
ucx_info -d
ucx_info -c | grep -E 'MUSA|IB_GPU_DIRECT'
ucx_info -f | grep -E 'MUSA|IB_GPU_DIRECT'
ucx_info -d | grep -A8 -E 'musa_cpy|musa_copy'
ucx_info -d | grep -A12 -E 'ib/'
```

Collect device, NIC, driver, and loader evidence:

```sh
musa-smi
ibv_devinfo
ibdev2netdev
ucx_info -d
ldconfig -p | grep -E 'libmusart|libmusa|libuc[sp]'
cat /proc/driver/musa/version 2>/dev/null || true
cat /sys/module/mt_peermem/version 2>/dev/null || true
cat /sys/module/mt_peer_mem/version 2>/dev/null || true
cat /sys/kernel/mm/memory_peers/mt_mem/version 2>/dev/null || true
```

The log must state the selected TLS and whether each transfer was `direct`
DMA-BUF, `mt-peermem`, or provider-internal staging. Bandwidth alone is not
evidence of a direct path.

## Contract tests

Run the SDK-free tests first:

```sh
make -C build-devel/test/gtest test \
    GTEST_FILTER='test_musa_contract.*'
```

The hardware test matrix must record pass/fail output and data checks for:

| Requirement | Positive cases | Negative cases |
| --- | --- | --- |
| U-01/U-02 | `musa` name, bitmap with runtime enabled | runtime disabled |
| U-03 | device allocation, HOST, interior pointer | invalid pointer, managed/VMM |
| U-04 | base plus offset, repeated registration | zero length, cross-boundary |
| U-05 | two processes, two nodes, packed IB rkey | stale key, peer exit |
| U-06 | D-D, H-D, D-H, H-H READ/WRITE | unsupported combination |
| U-07 | immediate, error pointer, async request | request leak |
| U-08 | flush/fence before notification | notification before data visible |
| U-09/U-10 | caller/background progress, worker per device | wrong current device |
| U-11 | peer/link failure, cancel, close | unbounded wait |
| U-12 | headers, SONAMEs, loader path | mixed ABI components |

U-03 must prove that the original MUSA allocation is registered and queried;
it must not pass a pointer relabeled as `host`. U-05 through U-11 require
reproducible logs from both processes and explicit timeout values.

## Two-process and two-node commands

Use a small test program that allocates ordinary device memory, fills it on
the producer stream, waits for that stream, posts the NIXL transfer, and
validates the destination after local completion. The command template is:

```sh
UCX_LOG_LEVEL=info \
UCX_TLS=musa_copy,ib,self \
UCX_NET_DEVICES=<hca>:<port> \
UCX_MUSA_REGISTRATION_MODE=auto \
./musa_ucx_e2e --role=server --listen=<addr>:<port> \
    --bytes=<size> --timeout-ms=5000

UCX_LOG_LEVEL=info \
UCX_TLS=musa_copy,ib,self \
UCX_NET_DEVICES=<hca>:<port> \
UCX_MUSA_REGISTRATION_MODE=auto \
./musa_ucx_e2e --role=client --connect=<addr>:<port> \
    --bytes=<size> --timeout-ms=5000
```

Run the pair on one host and then on two hosts. Capture both stdout/stderr
with timestamps, the selected transport, memory type, DMA-BUF fd/offset
metadata, data checksum, and final status. `MUSA_UCX` is the NIXL-side
selection name; this UCX module does not implement a NIXL provider.

## Acceptance rules

Initialization or registration must fail when MUSA cannot be identified,
when the runtime is absent, or when the selected direct path is unavailable.
`managed`, VMM, heterogeneous cards, IPC, and host staging are outside the
ordinary-device MVP unless their capability and limitations are separately
declared.

U-01, U-02, and U-03 are initialization gates. If any gate fails, NIXL
`MUSA_UCX` must refuse initialization or registration. Until U-05 through
U-11 pass on real hardware, report the result as a local experiment rather
than a network MVP acceptance.
