# UCX MUSA Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional MUSA 5.2.0 UCT module that reports real MUSA
capability, validates ordinary device allocations, exposes DMA-BUF metadata,
supports local MUSA copies, and lets existing UCP/IB RMA machinery perform
direct registration when `mt-peermem` or DMA-BUF is available.

**Architecture:** Keep MUSA SDK calls inside `src/uct/musa`. The module's MD
owns runtime detection, allocation-range validation, device-context selection,
DMA-BUF export, and lightweight local registration handles. Existing UCP
registration, rkey, progress, request, and failure paths remain authoritative;
the generic UCP DMA-BUF provider path supplies the exported fd to the IB MD.
The IB MD only adds MUSA peer-memory capability when a documented MUSA
peer-memory kernel path is visible.

**Tech Stack:** C, Autoconf/Automake, UCX UCT/UCP APIs, MUSA 5.2.0 runtime
(`libmusart`) and driver (`libmusa`), GoogleTest, existing IB verbs and
DMA-BUF APIs.

---

## File map

- Modify `src/ucs/memory/memory_type.h` and `.c` for the stable `musa` enum
  value, name, and description.
- Create `config/m4/musa.m4`; modify `configure.ac`,
  `src/uct/configure.m4`, and `src/uct/Makefile.am` for optional SDK
  detection and module registration.
- Create `src/uct/musa/configure.m4`, `Makefile.am`, and `ucx-musa.pc.in`
  for the loadable module.
- Create `src/uct/musa/musa_utils.h` for SDK-independent range and mode
  contracts used by tests and MUSA code.
- Create `src/uct/musa/musa_base.[ch]` for runtime initialization, visible
  devices, pointer attributes, allocation ranges, device restoration, and
  DMA-BUF export.
- Create `src/uct/musa/musa_md.[ch]` for MUSA MD attributes, memory query,
  detection, registration, deregistration, and registration-mode config.
- Create `src/uct/musa/musa_copy_iface.[ch]` and
  `src/uct/musa/musa_copy_ep.[ch]` for synchronous local H-H/H-D/D-H/D-D
  operations with one UCT worker per opened interface.
- Modify `src/uct/ib/base/ib_md.c` to detect MUSA peer-memory paths and add
  `UCS_MEMORY_TYPE_MUSA` to the existing IB registration capability.
- Create `test/gtest/uct/test_musa_contract.cc` for SDK-free type, mode, and
  overflow tests; add it to `test/gtest/Makefile.am`.
- Create `docs/musa_provider_validation.md` for fixed-build, topology, ABI,
  two-process, two-node, and evidence commands. Do not add generated build
  output or the local PDF to Git.

### Task 1: Write SDK-free failing contract tests

**Files:**
- Create: `test/gtest/uct/test_musa_contract.cc`
- Modify: `test/gtest/Makefile.am`
- Test helper interface to be added: `src/uct/musa/musa_utils.h`

- [ ] **Step 1: Add the test source before production implementation**

```cpp
#include <gtest/gtest.h>

#include <ucs/memory/memory_type.h>
#include <uct/musa/musa_utils.h>

TEST(test_musa_contract, memory_type_name_is_stable)
{
    EXPECT_STREQ(ucs_memory_type_names[UCS_MEMORY_TYPE_MUSA], "musa");
    EXPECT_NE(ucs_memory_type_descs[UCS_MEMORY_TYPE_MUSA], nullptr);
    EXPECT_EQ(UCS_MEMORY_TYPE_MUSA + 1, UCS_MEMORY_TYPE_LAST);
}

TEST(test_musa_contract, range_accepts_interior_subrange)
{
    EXPECT_TRUE(uct_musa_range_is_valid(0x1000, 0x1000, 0x1800, 0x400));
}

TEST(test_musa_contract, range_rejects_cross_allocation_subrange)
{
    EXPECT_FALSE(uct_musa_range_is_valid(0x1000, 0x1000, 0x1f00, 0x200));
}

TEST(test_musa_contract, range_rejects_integer_overflow)
{
    EXPECT_FALSE(uct_musa_range_is_valid(UINTPTR_MAX - 0x10, 0x20,
                                         UINTPTR_MAX - 0x10, 0x20));
}

TEST(test_musa_contract, registration_mode_values_are_distinct)
{
    EXPECT_NE(UCT_MUSA_REG_MODE_AUTO, UCT_MUSA_REG_MODE_DMABUF);
    EXPECT_NE(UCT_MUSA_REG_MODE_DMABUF, UCT_MUSA_REG_MODE_PEERMEM);
    EXPECT_NE(UCT_MUSA_REG_MODE_PEERMEM, UCT_MUSA_REG_MODE_STAGING);
}
```

Add `uct/test_musa_contract.cc` to `gtest_SOURCES` in the existing UCT source
list. The test includes only UCX headers and the SDK-free MUSA helper header.

- [ ] **Step 2: Run the focused test and verify the expected red failure**

Run:

```sh
make -C build-devel test/gtest/test GTEST_FILTER=test_musa_contract.*
```

Expected: compilation fails because `UCS_MEMORY_TYPE_MUSA` and
`uct/musa/musa_utils.h` do not exist. This confirms the test is exercising the
new contract rather than existing behavior.

- [ ] **Step 3: Add the minimal SDK-free helper and enum**

Create `src/uct/musa/musa_utils.h` with:

```c
typedef enum {
    UCT_MUSA_REG_MODE_AUTO,
    UCT_MUSA_REG_MODE_DMABUF,
    UCT_MUSA_REG_MODE_PEERMEM,
    UCT_MUSA_REG_MODE_STAGING,
    UCT_MUSA_REG_MODE_LAST
} uct_musa_reg_mode_t;

static UCS_F_ALWAYS_INLINE int
uct_musa_range_is_valid(uintptr_t base, size_t alloc_length,
                        uintptr_t address, size_t length)
{
    uintptr_t alloc_end;
    uintptr_t address_end;

    if ((alloc_length == 0) || (length == 0) || (address < base)) {
        return 0;
    }
    if (base > (UINTPTR_MAX - alloc_length)) {
        return 0;
    }
    alloc_end = base + alloc_length;
    if (address > (UINTPTR_MAX - length)) {
        return 0;
    }
    address_end = address + length;
    return address_end <= alloc_end;
}
```

Add `UCS_MEMORY_TYPE_MUSA` immediately before `UCS_MEMORY_TYPE_LAST`, add
`"musa"` and `"MUSA device memory"` to the name/description arrays, and do not
add it to `UCS_MEMORY_TYPES_CPU_ACCESSIBLE`.

- [ ] **Step 4: Run the focused test and verify green**

Run the same focused target. Expected: all five tests pass.

- [ ] **Step 5: Commit the contract slice**

```sh
git add src/ucs/memory/memory_type.h src/ucs/memory/memory_type.c \
        src/uct/musa/musa_utils.h test/gtest/Makefile.am \
        test/gtest/uct/test_musa_contract.cc
git commit -s -m "UCS/MEMORY: Add MUSA memory type contract"
```

### Task 2: Add MUSA SDK configure and module scaffolding

**Files:**
- Create: `config/m4/musa.m4`
- Create: `src/uct/musa/configure.m4`
- Create: `src/uct/musa/Makefile.am`
- Create: `src/uct/musa/ucx-musa.pc.in`
- Modify: `configure.ac`, `src/uct/configure.m4`, `src/uct/Makefile.am`

- [ ] **Step 1: Add the failing configure/build contract**

Add a configure-only check command to the validation notes and run the
existing tree before implementing the macro:

```sh
./configure --help | grep -- --with-musa
```

Expected: the option is absent. This is the red check for the new build
interface.

- [ ] **Step 2: Implement `UCX_CHECK_MUSA`**

`config/m4/musa.m4` must:

1. Accept `--with-musa=DIR`, with `guess` as the default and no probing when
   explicitly disabled.
2. For an explicit directory, set `MUSA_CPPFLAGS=-I$DIR/include`,
   `MUSA_LDFLAGS=-L$DIR/lib -L$DIR/lib64`, and
   `MUSA_LIBS="-lmusart -lmusa"`.
3. Check `musa_runtime_api.h`, `musa.h`, `musaGetDeviceCount`,
   `musaPointerGetAttributes`, `musaGetDevice`, `musaSetDevice`,
   `musaMemcpy`, `musaGetErrorString`, and `muMemGetAddressRange`.
4. Link a small program that uses the runtime and driver declarations.
5. Define `HAVE_MUSA`, substitute all MUSA variables, and enable
   `HAVE_MUSA` only after every check succeeds.
6. Fail configuration for an explicit invalid path; warn and disable MUSA
   for default autodetection failure.

The module does not link the DMA-BUF export symbol directly. It resolves
`muMemGetHandleForAddressRange` through `dlsym` because MUSA 5.2.0 deployments
may ship the driver symbol independently of the compile-time driver header.

- [ ] **Step 3: Wire the module into Autoconf and Automake**

Include `config/m4/musa.m4` in the normal configure branch and define
`AM_CONDITIONAL([HAVE_MUSA], [test "x$musa_happy" = xyes])` in both normal and
docs-only branches. Include `src/uct/musa/configure.m4` from
`src/uct/configure.m4`, add `musa` to `src/uct/Makefile.am` `SUBDIRS`, append
`:musa` to `uct_modules` only when the SDK checks pass, and add the generated
MUSA module Makefile/pkg-config files to `AC_CONFIG_FILES`.

The MUSA module Makefile must use `$(MUSA_CPPFLAGS)`, `$(MUSA_LDFLAGS)`,
`$(MUSA_LIBS)`, `libucs.la`, and `libuct.la`, and must be guarded by
`if HAVE_MUSA`. It lists every new header/source explicitly.

- [ ] **Step 4: Run the configure contract and verify green**

Run:

```sh
./autogen.sh
./configure --help | grep -- --with-musa
./configure --without-musa
```

Expected: the help line is present; `--without-musa` configures successfully
without compiling or loading a MUSA module.

- [ ] **Step 5: Commit the build slice**

```sh
git add config/m4/musa.m4 configure.ac src/uct/configure.m4 \
        src/uct/Makefile.am src/uct/musa/configure.m4 \
        src/uct/musa/Makefile.am src/uct/musa/ucx-musa.pc.in
git commit -s -m "UCT/BUILD: Add optional MUSA module detection"
```

### Task 3: Implement the runtime adapter and MUSA MD

**Files:**
- Create: `src/uct/musa/musa_base.h`, `src/uct/musa/musa_base.c`
- Create: `src/uct/musa/musa_md.h`, `src/uct/musa/musa_md.c`
- Modify: `src/uct/musa/Makefile.am`

- [ ] **Step 1: Add a MUSA-gated runtime test entry**

Add a real-SDK UCT MD test source guarded by `#ifdef HAVE_MUSA` only when the
module is enabled. The first case must query `uct_md_attr_v2_t` and assert:

```cpp
ASSERT_TRUE(md_attr.detect_mem_types & UCS_BIT(UCS_MEMORY_TYPE_MUSA));
EXPECT_EQ(ucs_memory_type_names[UCS_MEMORY_TYPE_MUSA], std::string("musa"));
```

The test skips with `UCS_TEST_SKIP_R` when no visible MUSA resource exists.
Before the implementation it must fail to build with the missing component and
MD symbols.

- [ ] **Step 2: Implement runtime initialization and device enumeration**

`musa_base.c` must:

- Call `musaGetDeviceCount` once under a mutex/initialization guard.
- Return an empty MD/TL resource list when the runtime is absent at load time,
  returns an error, or reports zero devices.
- Return one accelerator TL device per visible ordinal, named `musa0`,
  `musa1`, and so on.
- Never create a hidden progress thread.
- Log the original MUSA error string with the UCX operation name.

`musa_base.h` exposes:

```c
ucs_status_t uct_musa_base_query_md_resources(
        uct_component_h component, uct_md_resource_desc_t **resources_p,
        unsigned *num_resources_p);
ucs_status_t uct_musa_base_query_devices(
        uct_md_h md, uct_tl_device_resource_t **devices_p,
        unsigned *num_devices_p);
```

- [ ] **Step 3: Implement safe pointer and allocation-range queries**

`uct_musa_base_query_mem` must:

1. Reject null or zero-length requests.
2. Call `musaPointerGetAttributes`.
3. Return `UCS_ERR_INVALID_ADDR` for the runtime's
   `musaErrorInvalidValue` host/unregistered result; return an IO error for
   other runtime errors.
4. Accept only `musaMemoryTypeDevice`; return unsupported for managed or
   other non-device types.
5. Validate the device ordinal against the current visible device count.
6. Save the current device with `musaGetDevice`, select the pointer's device,
   call `muMemGetAddressRange`, and restore the saved device on every exit.
7. Use `uct_musa_range_is_valid` for base, allocation length, address, and
   requested length.
8. Return the allocation base, full allocation length, device ordinal, and
   unknown system device without updating any cache on error.

The adapter must use `UCS_ERR_INVALID_PARAM` for a cross-boundary or overflow
range and include the pointer range, device ordinal, operation, and MUSA error
string in diagnostics.

- [ ] **Step 4: Implement DMA-BUF export with ownership cleanup**

Resolve `muMemGetHandleForAddressRange` once with `dlsym(RTLD_DEFAULT, ...)`.
Use a function-pointer ABI equivalent to:

```c
typedef MUresult (*uct_musa_dmabuf_export_func_t)(
        void *handle, MUdeviceptr base, size_t length, int handle_type,
        unsigned long long flags);
```

Use handle type `0`, the MUSA 5.2.0 DMA-BUF-FD value, and export from the
allocation base, never from an interior pointer. Return the allocation-relative
offset separately. Close any fd after UCP/IB consumes it; close it immediately
when registration/export fails.

- [ ] **Step 5: Implement MD attributes and registration**

The MD query must report:

```c
flags              = UCT_MD_FLAG_REG;
reg_mem_types      = UCS_BIT(UCS_MEMORY_TYPE_MUSA);
cache_mem_types    = UCS_BIT(UCS_MEMORY_TYPE_MUSA);
access_mem_types   = UCS_BIT(UCS_MEMORY_TYPE_HOST) |
                     UCS_BIT(UCS_MEMORY_TYPE_MUSA);
detect_mem_types   = UCS_BIT(UCS_MEMORY_TYPE_MUSA);
required_mem_flags = UCS_MEM_FLAG_REGISTRABLE;
```

Set `dmabuf_mem_types` to MUSA only in `auto` or `dmabuf` mode when the
export symbol is available. In `dmabuf` mode, fail MD open if export is
unavailable. In `peermem` and `staging` mode, do not advertise DMA-BUF.

Registration allocates a handle containing the user address, allocation base,
allocation length, requested length, offset, and device ordinal. It validates
the MUSA type and range before allocation. Deregistration frees exactly that
handle. It does not serialize a process-local pointer as a remote key and
does not set `UCT_MD_FLAG_NEED_RKEY`; direct network keys come from the IB MD.

- [ ] **Step 6: Run MUSA-enabled unit/build checks**

With a MUSA 5.2.0 SDK:

```sh
../contrib/configure-devel --with-musa=/path/to/musa --prefix=$PWD/install
make -j$(getconf _NPROCESSORS_ONLN)
make -C test/gtest test GTEST_FILTER=test_musa_contract.*
```

Expected: configure reports MUSA support, the module links against
`libmusart` and `libmusa`, and the SDK-free plus MUSA MD contract tests pass.
On this host, record the exact missing-SDK limitation instead of claiming this
step passed.

- [ ] **Step 7: Commit the runtime/MD slice**

```sh
git add src/uct/musa
git commit -s -m "UCT/MUSA: Add runtime detection and registration MD"
```

### Task 4: Implement synchronous `musa_copy`

**Files:**
- Create: `src/uct/musa/musa_copy_iface.h`,
  `src/uct/musa/musa_copy_iface.c`
- Create: `src/uct/musa/musa_copy_ep.h`, `src/uct/musa/musa_copy_ep.c`
- Modify: `src/uct/musa/Makefile.am`

- [ ] **Step 1: Add the failing operation tests**

Add MUSA-gated UCT tests for H-H, H-D, D-H, and D-D one-segment PUT/GET,
including one test with a non-default device ordinal. Use `musaMalloc`,
`musaMemcpy`, and `musaFree`; select the TL with `musa_copy`. Before the
operation implementation, the tests must fail because the TL is absent.

- [ ] **Step 2: Implement pointer-kind selection**

For every operation, query both pointers with the runtime adapter. Treat the
runtime invalid-value result as host memory, reject managed/unsupported kinds,
select the device from the device pointer, save the current device, call
`musaSetDevice`, and restore it after the operation.

Use:

```c
memcpy(dst, src, length); /* H-H */
musaMemcpy(dst, src, length, musaMemcpyHostToDevice);
musaMemcpy(dst, src, length, musaMemcpyDeviceToHost);
musaMemcpy(dst, src, length, musaMemcpyDeviceToDevice);
```

Return `UCS_OK` only after the synchronous copy returns success. Translate
MUSA failures to `UCS_ERR_IO_ERROR` and log the MUSA error string. Return
`UCS_ERR_UNSUPPORTED` for `iovcnt != 1` and invalid managed-memory cases.
Ignore `rkey` because this TL is process-local; direct remote operations use
the IB TL and normal packed keys.

- [ ] **Step 3: Implement the UCT interface and lifecycle**

The iface must advertise GET/PUT short and zcopy, pending, and connect-to-iface
flags; use one fixed device address containing the MUSA ordinal; provide base
fence and immediate flush; return zero from progress; and delete all
endpoints/interfaces through the existing UCX class lifecycle. It must not
start a background progress thread or retain caller stream state.

- [ ] **Step 4: Run operation tests and the focused suite**

Run the MUSA-gated operation tests on MUSA hardware and always run:

```sh
make -C test/gtest test GTEST_FILTER=test_musa_contract.*
```

Expected: immediate local copies pass, MUSA errors return UCX errors, and
worker teardown leaves no allocated handle or endpoint.

- [ ] **Step 5: Commit the copy slice**

```sh
git add src/uct/musa test/gtest
git commit -s -m "UCT/MUSA: Add synchronous copy transport"
```

### Task 5: Extend IB peer-memory capability and validate direct registration

**Files:**
- Modify: `src/uct/ib/base/ib_md.c`
- Modify: `src/uct/ib/base/ib_md.h` only if a declaration is needed
- Modify: `docs/musa_provider_validation.md`
- Test: `test/gtest/uct/ib/test_ib_md.cc` for capability assertions

- [ ] **Step 1: Add the failing MUSA capability test**

Add a test that queries an IB MD and asserts MUSA is in `reg_mem_types` only
when one of the MUSA peer-memory paths is present, and that
`UCT_MD_FLAG_REG_DMABUF` remains the separate DMA-BUF capability. Without the
IB implementation, the new conditional assertion fails on a matching host.

- [ ] **Step 2: Add MUSA peer-memory detection**

In `uct_ib_md_open_common`, under `#ifdef HAVE_MUSA` and the existing
GPU-direct configuration gate, check all supported MUSA peer-memory locations:

```c
uct_ib_check_gpudirect_driver(md, "/sys/module/mt_peermem/version",
                              UCS_MEMORY_TYPE_MUSA);
uct_ib_check_gpudirect_driver(md, "/sys/module/mt_peer_mem/version",
                              UCS_MEMORY_TYPE_MUSA);
uct_ib_check_gpudirect_driver(md,
                              "/sys/kernel/mm/memory_peers/mt_mem/version",
                              UCS_MEMORY_TYPE_MUSA);
```

Retain the existing generic `ibv_reg_mr` path. Because UCP already passes a
DMA-BUF fd from the MUSA MD into any MD with `UCT_MD_FLAG_REG_DMABUF`, the IB
code does not need a second MUSA export implementation or a new rkey format.
Extend the GPUDirect error diagnostic to mention MUSA peer memory and MUSA
DMA-BUF.

- [ ] **Step 3: Run direct-path tests**

On a MUSA GPU/NIC host, run:

```sh
UCX_TLS=rc,sm,self,musa_copy ucx_info -d
UCX_TLS=rc,sm,self ucx_info -d
make -C test/gtest test GTEST_FILTER=test_ib_md.*
```

Then run the two-process H-D, D-H, and D-D READ/WRITE program from the
validation document in `peermem`, `dmabuf`, and `auto` modes. Record the
selected mode, fd path, MR/rkey details without serializing pointer values as
keys, remote checksum, notification ordering, and cleanup result.

- [ ] **Step 4: Commit the IB slice**

```sh
git add src/uct/ib/base/ib_md.c test/gtest/uct/ib/test_ib_md.cc \
        docs/musa_provider_validation.md
git commit -s -m "UCT/IB: Detect MUSA peer-memory registration"
```

### Task 6: Add validation documentation and complete verification

**Files:**
- Create or modify: `docs/musa_provider_validation.md`
- Modify: `docs/superpowers/plans/2026-10-09-ucx-musa-provider.md`

- [ ] **Step 1: Document reproducible evidence commands**

Document SDK 5.2.0 prefix, UCX commit, compiler, `ucx_info -v/-d`, `ldd`
paths/SONAMEs, actual TLS list, MUSA-visible device ordinals, GPU/NIC
topology, peer-memory module/version, DMA-BUF capability, timestamps, and
commands for same-host and two-node READ/WRITE. Include explicit labels:
`MUSA memory-type support`, `MUSA staging path`, `MUSA direct-RDMA
experiment`, and `MUSA direct-RDMA network MVP accepted`.

- [ ] **Step 2: Run host-only verification**

Run and capture fresh output:

```sh
git diff --check
./autogen.sh
mkdir -p build-devel
cd build-devel
../contrib/configure-devel --without-musa --prefix=$PWD/install
make -j$(getconf _NPROCESSORS_ONLN)
make -C test/gtest test GTEST_FILTER=test_musa_contract.*
```

Expected: clean configure/build, all SDK-free MUSA contract tests pass, and
the MUSA module is absent from the no-SDK build. If an existing local build
directory is reused, inspect its configure flags and do not attribute stale
MUSA artifacts to this branch.

- [ ] **Step 3: Run optional hardware verification where available**

Run the MUSA-enabled unit tests, UCP capability and memory-map tests, UCT
worker/thread-mode tests, IB rkey pack/unpack tests, bounded peer-failure and
cancel tests, and the two-process/two-node READ/WRITE matrix. A missing MUSA
SDK, GPU, NIC, TLS, or peer-memory driver is recorded as unavailable
hardware evidence, never as a passing test.

- [ ] **Step 4: Re-read the acceptance boundary**

Confirm U-01 through U-04 from host/fake/runtime/range evidence. Claim only
`MUSA memory-type support` or `MUSA staging path` without hardware U-05
through U-11 evidence. Claim `MUSA direct-RDMA network MVP accepted` only
after the requested direct mode passes two-process and two-node ordering,
failure, cleanup, and ABI checks.

- [ ] **Step 5: Final verification before any completion claim**

Run:

```sh
git status --short --branch
git diff --check HEAD~1
make -C build-devel test/gtest test GTEST_FILTER=test_musa_contract.*
```

Read the full exit status and test counts. Do not state that the provider or
hardware MVP is complete when the MUSA SDK or hardware validation is absent.

## Execution record

- Contract-first red/green cycle completed with the SDK-free smoke test:
  the first compile failed because the new MUSA helper was absent, and the
  implemented enum/range helper then compiled and ran successfully.
- Contract, build-gate, and implementation slices were committed on
  `feat/musa_support`.
- The MUSA sources were syntax-checked against a temporary SDK declaration
  set covering the MUSA 5.2.0 runtime/driver calls used by this branch.
- The actual host is macOS/AArch64 without the MUSA SDK, Linux IB hardware, or
  an autotools `autoreconf` toolchain. Full configure, build, and U-05 through
  U-11 hardware validation therefore remain pending and are not claimed.
