# UCXX CI scripts

The scripts in this directory are run by UCXX's GitHub Actions workflows (see
`.github/workflows/`), using the RAPIDS CI images (`rapidsai/ci-conda`,
`rapidsai/ci-wheel`, `rapidsai/citestwheel`) and the `rapids-*` helpers they provide.

## Building and testing against a non-conda/non-wheel UCX (`UCXX_NO_LIBUCX`)

By default, UCXX conda packages depend on the conda-forge `ucx` package and UCXX wheels
depend on the `libucx-cu{12,13}` wheels. External projects that want to run UCXX's CI
against their own UCX installation (for example, a UCX built from source) can set:

```bash
export UCXX_NO_LIBUCX=true
```

and run the CI scripts in this directory unmodified. When unset, behavior is unchanged.

> [!IMPORTANT]
> This is intended for **testing UCXX against a specific UCX installation** in a
> controlled environment. Packages built this way have no dependency on UCX at all and
> only work on machines where a compatible UCX is installed exactly as it was at build
> time. They must not be published.

### What the switch does

| Component | Effect of `UCXX_NO_LIBUCX=true` |
|---|---|
| `dependencies.yaml` | Every `rapids-dependency-file-generator` call in these scripts passes `no_libucx=true`, selecting the existing empty `no_libucx` matrix entries, so no `ucx` / `libucx-cu*` requirement is generated. |
| `build_wheel_libucxx.sh` | Exports `RAPIDS_MATRIX_ENTRY="cuda_suffixed=true;no_libucx=true"` so `rapids-build-backend` omits `libucx-cu*` from the wheel's build and runtime dependencies. |
| `conda/recipes/{libucxx,ucxx}` | Drop `ucx` from `host`/`run` requirements and allowlist `libucp`, `libucs`, `libuct` and `libucm` in the overlinking check (`missing_dso_allowlist`). |

The value must be exactly `true`. Any other value (`1`, `yes`, `True`, ...) is silently
treated as unset. Set it consistently for **all** jobs (builds and tests).

### Locating the UCX installation

Both conda and wheel builds locate UCX with CMake's `find_package(ucx)`. It is found if
any of the following applies, in order of precedence:

1. **The `ucx_ROOT` environment variable is set to the UCX prefix** (recommended), e.g.,
   `export ucx_ROOT=/opt/hpcx/ucx`. Conda recipes forward it into their isolated build
   environments; wheel builds pass the environment to CMake directly. It must be an
   environment variable: `build_wheel_libucxx.sh` overwrites `SKBUILD_CMAKE_ARGS`, so
   `-Ducx_ROOT=...` cannot be passed that way. `UCX_ROOT` works as well, but `ucx_ROOT`
   is preferred and takes precedence if both are set: UCX treats `UCX_*` variables as
   its own configuration and prints `unused environment variable: UCX_ROOT` warnings in
   every process that initializes UCX, including test jobs (suppress with
   `UCX_WARN_UNUSED_ENV_VARS=n`).
2. **`<prefix>/bin` is on `PATH`.** CMake also searches the parent of every `bin`
   directory on `PATH`, and `rattler-build` preserves the caller's `PATH`. Since conda test
   jobs need `ucx_info` on `PATH` anyway, this often applies without further setup.
3. **UCX is in one of CMake's default prefixes**: `/usr/local`, `/usr` (wheels only, see
   [Conda limitations](#conda-limitations)), or a directory matching `/opt/ucx*` (CMake
   searches `/opt/<name>*`).

If no UCX is found, builds fail at CMake configure time with
`Could not find a package configuration file provided by "ucx"`. If more than one UCX
installation is discoverable, set `ucx_ROOT` to make the choice explicit.

UCX installed with prefix `/usr` does not work for conda builds, even with `ucx_ROOT`
(see [Conda limitations](#conda-limitations)).

### Requirements on the UCX installation

- **Discoverable by CMake**, see [Locating the UCX installation](#locating-the-ucx-installation).
- **Runtime:** UCX libraries must be found by the dynamic loader without help from UCXX,
  i.e., registered with `ldconfig` (e.g., via `/etc/ld.so.conf.d/`) or present in
  `LD_LIBRARY_PATH`. Neither conda packages nor wheels record UCX's location.
  `LD_LIBRARY_PATH` is not passed into `rattler-build`'s isolated environments; the CI
  test jobs inherit it normally. Conda builds load UCX while building `ucxx-tests`, so if
  UCX is not registered with `ldconfig`, `ucx_ROOT` (or `UCX_ROOT`) must be set: the
  build then adds `<prefix>/lib` and `<prefix>/lib64` to its own `LD_LIBRARY_PATH`.
  Package import tests are skipped regardless (see [Conda limitations](#conda-limitations)).
- **`ucx_info` must be on `PATH`** in conda test jobs (`test_cpp.sh`, `test_python.sh`),
  which call it unconditionally to print the UCX configuration. Wheel test jobs fall back to querying the version from Python.
- **For conda builds, UCX should be built against glibc 2.28 or older** (the conda
  sysroot version). See [Conda limitations](#conda-limitations).

### Conda limitations

Conda packages are built by `rattler-build` with the conda-forge compiler toolchain,
which compiles against its own sysroot rather than the host system. This leads to the
following:

1. **UCX installed with prefix `/usr` does not work**, regardless of how it is located.
   CMake finds `/usr/lib*/cmake/ucx`, but always strips `/usr/include` from compile lines
   because it is an implicit include directory (`Platform/UnixPaths.cmake`). The conda
   compiler does not search the host's `/usr/include`, so the build fails with
   `fatal error: ucp/api/ucp.h: No such file or directory`. Install UCX in another prefix
   (e.g., `/usr/local` or `/opt/...`).
2. **`CMAKE_PREFIX_PATH` cannot be used to locate UCX.** `rattler-build` runs builds in
   isolated environments that only receive variables forwarded by the recipes, and the
   conda compiler activation overwrites `CMAKE_PREFIX_PATH`. Use `ucx_ROOT` instead.
3. **No UCX version constraint.** The `ucx_min`/`ucx_max` bounds in
   `conda_build_config.yaml` are not applied, and CMake does not require a specific UCX
   version, so any UCX found is accepted at build time. This includes UCX 1.22.0, which
   regular builds exclude because it can segfault
   ([#783](https://github.com/rapidsai/ucxx/issues/783)). Only UCX 1.20.0 was tested.
4. **No UCX location is recorded.** `rattler-build` removes rpaths pointing outside the
   conda prefix (e.g., `/usr/local/lib64`), so UCX must be found by the loader at runtime
   (see [Requirements](#requirements-on-the-ucx-installation)).
5. **A conda `ucx` package can silently take precedence.** The built packages do not
   depend on or conflict with conda's `ucx`. If anything else in the environment pulls in
   `ucx`, it is installed alongside, and `libucxx.so` will likely resolve `libucp.so.0`
   from the conda environment (searched via `$ORIGIN` rpath) instead of the system
   installation (untested). Verify with `ucx_info -v` and
   `ldd $CONDA_PREFIX/lib/libucxx.so` in test logs.
6. **glibc compatibility is not checked.** Packages declare `__glibc >=2.28`, matching
   the conda sysroot. A system UCX built against a newer glibc may fail to link, or yield
   packages whose real glibc requirement is higher than declared (untested).
7. **Package import tests are skipped.** The `ucxx` recipe runs
   `import ucxx` tests in isolated `rattler-build` environments,
   which cannot load UCX unless it is registered with `ldconfig` (unlike builds, tests
   do not receive `ucx_ROOT`/`UCX_ROOT`). These tests are skipped
   with `UCXX_NO_LIBUCX=true`; imports are exercised by `test_python.sh` instead.
   Other package tests (e.g., file presence) still run.

### Wheel limitations

Wheels are built with the CI image's system compiler (gcc-toolset), not a sysroot, so the
`/usr` include problem above does not apply.

1. **The wheels are not portable.** They carry manylinux tags but require a system UCX at
   runtime; `auditwheel repair` already excludes `libucp.so.0`, so UCX is not bundled.
2. **No UCX version constraint**, as for conda (including the excluded UCX 1.22.0).
3. With UCX at prefix `/usr`, only the `libucxx` wheel build was verified; the `ucxx`
   wheel build and wheel tests were verified with UCX at `/usr/local` and `/opt/ucx` only.

### Test environment notes

These are not specific to `UCXX_NO_LIBUCX`, but are easy to hit when reproducing CI
outside GitHub Actions:

- `rapidsai/ci-wheel` images enable the required compiler (gcc-toolset) via
  `/etc/profile.d/enable_devtools.sh`, which is only sourced by login shells. Without it,
  the system's GCC is used and the C++ build fails.
- Scripts download artifacts from earlier jobs with `rapids-download-from-github` and
  configure S3-backed sccache with `rapids-configure-sccache`; outside GitHub Actions these
  need to be replaced or stubbed.
