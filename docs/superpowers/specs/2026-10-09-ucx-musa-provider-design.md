# UCX MUSA Provider MVP System Design

Date: 2026-10-09

Branch: `feat/musa_support`

Target SDK: MUSA 5.2.0

Status: approved design; implementation and hardware acceptance are pending.

## 1. Scope

This design adds MUSA-aware memory handling to UCX for ordinary MUSA device
allocations. The provider is a UCX UCT memory-domain and transport integration.
It is not a Libfabric provider, and NIXL remains a UCP consumer.

The implementation is direct-first:

1. Use existing UCX IB registration and RMA machinery when MUSA peer memory or
   DMA-BUF registration is available.
2. Use a separate `musa_copy` UCT path for local copies and explicitly selected
   host staging.
3. Report the selected path through UCX component/TL diagnostics and timestamped
   failure logs.

The implementation must never report a MUSA device allocation as host, CUDA, or
ROCm memory.

## 2. Goals

The MVP provides:

- A stable UCX memory type named `musa`.
- Runtime capability gating based on MUSA 5.2.0 availability and visible
  devices.
- Automatic detection of ordinary MUSA device allocations.
- Registration of interior subranges with allocation-base and length checks.
- Direct IB registration through `mt-peermem` when the peer-memory path is
  present.
- Direct IB registration through DMA-BUF when both MUSA export and IB
  DMA-BUF registration are available.
- Existing UCP PUT/GET, rkey, flush, fence, worker progress, and request
  lifetime semantics.
- Explicit synchronous local MUSA copies and an observable host-staging
  fallback.
- SDK-free tests for type contracts and fake-runtime tests for detection and
  range handling.

## 3. Non-goals

The MVP does not claim support for:

- Managed or unified allocations.
- VMM allocations unless the ordinary-device contract is independently
  verified for the allocation.
- MUSA IPC handles.
- Heterogeneous device memory.
- A new network protocol, rkey wire format, or UCP request state machine.
- A new NIXL provider API.
- Implicit staging presented as GPU-direct.
- Hardware certification without MUSA GPU, NIC, driver, and two-process
  evidence.

## 4. Requirement Mapping

| Requirement | Design coverage |
| --- | --- |
| U-01 | Add `UCS_MEMORY_TYPE_MUSA` and name it `musa`. |
| U-02 | The UCP bitmap includes MUSA only when the MUSA UCT module exposes a live device capability. |
| U-03 | MUSA MD queries pointer attributes and allocation extent; invalid or unsupported MUSA queries fail without reclassification. |
| U-04 | Base, allocation length, offset, and requested length are validated before registration. |
| U-05 | Direct IB uses normal IB memory keys, which contain registration metadata rather than process-local pointers. Staging uses host IB keys. |
| U-06 | UCP continues to select ordinary RMA protocols; direct paths use IB and staging paths use the existing UCP memtype-copy flow. |
| U-07 | UCP immediate, error-pointer, and asynchronous request contracts remain unchanged. Local MUSA copies complete synchronously. |
| U-08 | Existing UCT fence/flush ordering is preserved. Direct visibility is a hardware acceptance condition; staging completes the MUSA copy before reporting local completion. |
| U-09 | Existing UCP/UCT worker thread and wakeup behavior is reused. No hidden progress thread is added. |
| U-10 | Every MUSA runtime operation restores the device selected from pointer attributes on the calling worker or registration thread. |
| U-11 | Existing UCP peer-error, cancel, close, and flush handling is reused; MUSA calls return bounded errors and never spin on failure. |
| U-12 | Configure probes, build metadata, runtime paths, SONAMEs, and `ucx_info` evidence are recorded as one ABI set. |

U-05 through U-11 are not considered network MVP acceptance until the
hardware-gated tests pass. A CPU-only or fake-runtime result is provider
contract evidence, not direct-RDMA certification.

## 5. Architecture

```text
Application or NIXL
        |
        v
      UCP
        |
        +--> MUSA MD: detect type, extent, device, DMA-BUF metadata
        |
        +--> IB MD/TL: peer-memory MR or DMA-BUF MR, normal rkey/RMA
        |
        +--> musa_copy TL: local copy or explicit host staging
        |
        v
 MUSA runtime 5.2.0 and driver
        |
        +--> libmusart runtime calls
        +--> libmusa driver calls
        +--> mt-peermem or DMA-BUF kernel path
```

The MUSA SDK is included only in MUSA-owned compilation units. Generic UCP
code consumes the existing `ucs_memory_type_t` and UCT MD attributes. No
CUDA compatibility macro or ROCm alias is introduced.

## 6. Public Memory-Type Contract

Add `UCS_MEMORY_TYPE_MUSA` immediately before `UCS_MEMORY_TYPE_LAST`, preserving
the numeric values of existing memory types. Add:

- `ucs_memory_type_names[UCS_MEMORY_TYPE_MUSA] = "musa"`.
- A description identifying MUSA device memory.
- The MUSA bit to all relevant memory-type iteration and bitmap paths through
  the existing `UCS_MEMORY_TYPE_LAST` boundary.

MUSA device memory is not CPU-accessible. It must not be added to
`UCS_MEMORY_TYPES_CPU_ACCESSIBLE`.

`ucp_context_query(UCP_ATTR_FIELD_MEMORY_TYPES)` obtains its bitmap from the
existing UCP resource scan. When the SDK is absent, no device is visible, or
the MUSA module cannot initialize, the MUSA bit is absent. No environment
variable may rewrite MUSA as `host`.

## 7. Build and Capability Gating

Add an optional Autoconf check:

```text
--with-musa=DIR
```

The explicit directory must contain the MUSA 5.2.0 headers and libraries. The
probe must compile and link the symbols actually used by the provider,
including:

- `musa_runtime_api.h`.
- `musaGetDeviceCount`.
- `musaPointerGetAttributes`.
- `musaSetDevice`.
- `musaMemcpy`.
- `musaGetErrorString`.
- The driver allocation-range and DMA-BUF export APIs used by the selected
  implementation.

The runtime library is `libmusart`; the driver library is `libmusa`. Linking
only a driver library is insufficient for runtime capability. An explicit,
invalid `--with-musa=DIR` fails configuration with a diagnostic. Default
configuration without MUSA keeps the existing build unchanged.

The MUSA module is built as an optional UCX module and is loaded only when the
runtime reports at least one visible device. The module reports an empty
resource list when the SDK or runtime is unavailable.

The build records:

- UCX fixed source commit.
- MUSA SDK 5.2.0 version and prefix.
- Compiler and linker flags.
- `libucp`, `libucs`, `libuct`, and MUSA module paths.
- SONAME and dependency output from `ldd` or the platform equivalent.

## 8. Memory Detection and Registration

### 8.1 Detection

The MUSA MD uses the runtime pointer-attribute API to distinguish:

- Ordinary MUSA device allocation: `UCS_MEMORY_TYPE_MUSA`.
- Host allocation or unregistered host pointer: host memory.
- Managed, VMM, unsupported, or invalid pointer: an explicit unsupported or
  I/O error according to the runtime result.

For a device pointer, the provider obtains the allocation base and full
allocation length through the MUSA driver allocation-range API. The requested
address and length must satisfy:

```text
base <= address
address + length <= base + allocation_length
```

All additions and subtractions are overflow-safe. A failed attribute or range
query does not update the memory-type cache and does not fall back to host.

### 8.2 Registration

The MUSA MD advertises:

- Detection of `UCS_MEMORY_TYPE_MUSA`.
- Registration and access for host and MUSA memory where the chosen path
  supports it.
- DMA-BUF metadata only when the MUSA export API and runtime probe succeed.
- Registration flags that prevent UCP from treating unsupported MUSA memory as
  CPU-accessible.

For a subrange, the registration record retains:

- User address and requested length.
- Allocation base and full allocation length.
- Offset from allocation base.
- MUSA device ordinal.
- Selected registration mode.

Deregistration releases exactly the resource acquired by that registration.
Repeated registration of the same subrange is compatible with the existing
UCX registration-cache rules. A failed registration never publishes a partial
memory handle.

### 8.3 DMA-BUF

When DMA-BUF is selected, the provider:

1. Queries the allocation base and extent.
2. Exports a DMA-BUF from the allocation base using the MUSA driver API.
3. Computes the offset from allocation base to the user address.
4. Passes the fd and offset to the existing UCT MD registration parameters.
5. Closes the fd after successful `ibv_reg_dmabuf_mr` ownership transfer or
   immediately on failure, according to the verbs contract.

The provider must not export a user interior pointer as if it were an
allocation base. It must not use a CUDA page-size assumption for MUSA.

## 9. Direct IB Paths

### 9.1 Peer-memory registration

When the documented `mt-peermem` kernel path is present, the IB MD adds MUSA to
its registrable memory types. It uses the existing `ibv_reg_mr` flow. The
provider records the detected peer-memory path in diagnostics.

If the peer-memory path is absent, `ibv_reg_mr` is not attempted for MUSA
memory and the provider reports the reason.

### 9.2 DMA-BUF registration

When both MUSA export and `ibv_reg_dmabuf_mr` are available, the IB MD can
register a MUSA allocation through DMA-BUF. The existing IB packed rkey is
used; no MUSA pointer is serialized as a remote key.

If a NIC or driver rejects DMA-BUF, the provider may select peer memory only
when that path was independently detected. It must not report a successful
direct path after both mechanisms fail.

### 9.3 Mode selection

The provider exposes a configuration mode with the following semantics:

| Mode | Behavior |
| --- | --- |
| `auto` | Prefer DMA-BUF, then peer memory, then explicit staging if enabled. |
| `dmabuf` | Require MUSA DMA-BUF export and IB DMA-BUF registration. |
| `peermem` | Require `mt-peermem` and ordinary IB registration. |
| `staging` | Do not claim direct registration; use the `musa_copy`/UCP staging path. |

`dmabuf` and `peermem` modes fail initialization when their required
capability is missing. `auto` reports the selected mode in `ucx_info` and
diagnostic logs.

## 10. `musa_copy` and Staging

The `musa_copy` component is a separate UCT module backed by the MUSA runtime.
It is responsible only for local copies and the explicit host-staging path.

It supports:

- Host to host through a normal CPU copy.
- Host to MUSA through `musaMemcpy`.
- MUSA to host through `musaMemcpy`.
- MUSA to MUSA through `musaMemcpy`.

The copy operation is synchronous for the initial MVP. Returning `UCS_OK`
means the local copy is complete. Runtime errors return a UCX error status and
include the original MUSA error string in the log. Fence and flush for this
transport are no-ops only because the copy has completed before return.

UCP's existing memtype protocol may use this TL to stage a network operation
through host memory. Diagnostics must identify this as staging. A staging
success is not direct GPU-direct evidence.

## 11. Synchronization and Threading

The application remains responsible for completing the producer MUSA stream
before posting a transfer. The provider does not synchronize arbitrary
application streams.

For provider-owned runtime calls:

- Query pointer attributes before selecting the device.
- Call `musaSetDevice` on the current worker or registration thread.
- Do not depend on a previous caller-side device selection.
- Restore or preserve the thread's prior device according to the MUSA runtime
  contract.

Network progress, wakeup, request allocation, cancellation, and worker thread
levels remain UCP responsibilities. The MUSA module does not create a hidden
progress thread. Direct network completion and UCP flush remain the ordering
boundary for remote visibility. Hardware acceptance must verify that a remote
notification never observes stale data.

## 12. Error and Cleanup Rules

- Missing SDK or runtime: MUSA capability absent or explicit initialization
  failure.
- Missing TLS or NIC registration path: report the actual path and fail the
  requested direct mode.
- Pointer attribute failure: return an error or unsupported status; never
  classify it as host.
- Range overflow or cross-allocation registration: return invalid parameter.
- DMA-BUF export failure: close any acquired fd and return the original error.
- MR registration failure: release the fd and provider-side handle exactly once.
- Rkey packing or unpacking failure: use existing UCT cleanup and never expose
  a process-local pointer as a key.
- Peer or transport failure: reuse existing UCP error handling and bounded
  flush/cancel behavior.

Logs include operation, pointer range, MUSA device ordinal, registration mode,
UCX component/TL, original MUSA error, and timestamp. They do not include
payload data or credentials.

## 13. TDD and Verification Plan

Implementation follows red, green, and focused regression cycles.

### 13.1 SDK-free tests

Write before implementation:

- Stable memory-type enum/name/description checks.
- MUSA bit absent from a build without the module.
- Overflow-safe range helper tests.
- Mode selection tests for `auto`, `dmabuf`, `peermem`, and `staging`.
- Cleanup ownership tests for fd and registration handles.

### 13.2 Fake-runtime tests

Compile the production MUSA adapter against a small fake SDK and test:

- Nonzero and zero device counts.
- Ordinary device pointer classification.
- Host pointer classification.
- Managed and unsupported pointer rejection.
- Runtime error and invalid-pointer handling.
- Nonzero device ordinal and device-context selection.
- Allocation base, extent, interior offset, and cross-boundary rejection.
- DMA-BUF export failure and fd cleanup.

### 13.3 UCX tests

Add focused UCT/UCP tests for:

- `ucp_context_query` capability bitmap.
- `ucp_mem_map` and `ucp_mem_query` with a real MUSA allocation when hardware
  is available.
- Repeated subrange registration and deregistration.
- Direct D-D, H-D, D-H, and H-H operations.
- Immediate completion and asynchronous network request paths.
- Rkey pack/unpack through an IB MD without pointer-only keys.
- Flush/fence ordering before notification.
- Caller-progress and thread-mode operation.
- Peer failure, cancel, close, and bounded timeout behavior.

Tests requiring a MUSA SDK, GPU, or NIC are explicitly hardware-gated and fail
when requested hardware is missing; they are not successful skips.

### 13.4 Evidence commands

The release checklist records output from:

```text
git rev-parse HEAD
ucx_info -v
ucx_info -d
ldd libucp.so
ldd libucs.so
ldd libuct.so
lsmod
GPU and NIC topology tools
```

Each two-process or two-node test records both command lines, timestamps,
selected mode, TLS list, buffer type, data checksum, and cleanup result.

## 14. File Boundaries

Expected implementation areas:

- `src/ucs/memory/memory_type.h` and `.c`: public type and names.
- `config/m4/musa.m4`, `src/uct/configure.m4`, and `configure.ac`: SDK checks.
- `src/uct/musa/`: MUSA runtime, MD, copy iface, endpoint, and module build.
- `src/uct/ib/base/ib_md.c`: MUSA peer-memory capability detection.
- `src/uct/Makefile.am` and MUSA module `Makefile.am`: build ownership.
- `test/gtest/ucs`, `test/gtest/uct`, and `test/gtest/ucp`: focused tests.
- `docs/musa_provider_validation.md`: fixed-build and hardware evidence
  checklist, kept separate from the design.

No UCP protocol rewrite, CUDA code rename, ROCm alias, NIXL API change, or
generated build output belongs in this change.

## 15. Acceptance Boundary

The branch may claim a MUSA-aware UCX provider only when U-01 through U-04
pass the SDK-free, fake-runtime, and available hardware tests.

It may claim a direct-RDMA network MVP only when the requested peer-memory or
DMA-BUF mode passes two-process and two-node READ/WRITE data validation,
flush/notification ordering, thread-mode tests, failure cleanup, and ABI
evidence.

Until then, reports must use one of these precise labels:

- `MUSA memory-type support`: type and runtime detection only.
- `MUSA staging path`: host staging is selected and observable.
- `MUSA direct-RDMA experiment`: direct registration works in a named tested
  environment but the full U-01 through U-12 matrix is incomplete.
- `MUSA direct-RDMA network MVP accepted`: all required evidence is present.
