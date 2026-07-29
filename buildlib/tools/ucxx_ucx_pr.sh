# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#
# Sourced by build_ucxx.sh / test_ucxx.sh. Provides:
#   build_ucx_pr_conda  - build the PR's UCX into UCX_PR_PREFIX via a
#                         conda-forge toolchain env
#   use_pr_ucx          - patch the ucxx tree (cwd) so every package build and
#                         test consumes the PR's UCX instead of released ucx
# Requires: ucx_dir, UCX_PR_PREFIX exported by the caller; cwd = UCXX_DIR for
# use_pr_ucx.

# Runs inside the toolchain env created by build_ucx_pr_conda (invoked there by
# name through conda run). Idempotent - the container, and thus /tmp, is shared
# by all steps of a job.
build_ucx_pr() {
  [ -x "$UCX_PR_PREFIX/bin/ucx_info" ] && return 0
  # CUDA comes from the env's conda-forge cuda dev packages, pinned to
  # RAPIDS_CUDA_VERSION; headers and stubs live under targets/<arch>-linux.
  local cuda_opt="" conda_cuda="${CONDA_PREFIX:-}/targets/$(uname -m)-linux"
  [ -e "$conda_cuda/include/cuda.h" ] && cuda_opt="--with-cuda=$conda_cuda"
  echo "UCX-PR configure cuda: ${cuda_opt:-NONE (no cuda headers found)}"
  (cd "$ucx_dir" \
   && ./autogen.sh \
   && ./contrib/configure-release --prefix="$UCX_PR_PREFIX" $cuda_opt \
        --enable-mt --without-java --without-go --disable-doxygen-doc \
   && make -j"$(nproc)" install) > /tmp/ucx-pr-build.log 2>&1 \
    || { tail -50 /tmp/ucx-pr-build.log >&2; echo "ERROR: UCX (PR) build failed" >&2; return 1; }
}
export -f build_ucx_pr

# The conda image has no system toolchain; build the PR's UCX with a
# conda-forge one. Everything version-sensitive is pinned, because the env is
# solved fresh on every run and conda-forge moves under it: gcc 15+ turns the
# OpenMP 'master' deprecation into an error under UCX's -Werror, newer libtool
# releases break UCX's autogen, and sysroots past 2.17 stamp x86-64-v3 ISA
# notes that the pre-v3 CPUs of the GPU nodes refuse to load.
build_ucx_pr_conda() {
  [ -x "$UCX_PR_PREFIX/bin/ucx_info" ] && return 0
  local sysroot_pkg="sysroot_linux-64"
  [ "$(uname -m)" = "aarch64" ] && sysroot_pkg="sysroot_linux-aarch64"
  rapids-mamba-retry create -y -n ucx-build -c conda-forge \
    gcc=14 gxx=14 binutils make autoconf automake libtool=2.4.7 "$sysroot_pkg=2.17" \
    cuda-cudart-dev cuda-driver-dev cuda-nvml-dev cuda-crt cuda-cccl \
    "cuda-version=${RAPIDS_CUDA_VERSION%.*}" \
    > /tmp/ucx-toolchain.log 2>&1 \
    || { tail -30 /tmp/ucx-toolchain.log >&2; echo "ERROR: toolchain env create failed" >&2; return 1; }
  conda run -n ucx-build bash -ec build_ucx_pr \
    || { echo "ERROR: UCX (PR) build failed in toolchain env" >&2; return 1; }
}

# Patches the ucxx tree (cwd) so packages build against and run on the PR's
# UCX: strips the released ucx/libucx dependencies from the recipes and
# dependencies.yaml, relaxes the linker checks accordingly, and points the
# builds at UCX_PR_PREFIX. Idempotent; every patch is guarded fail-loud.
use_pr_ucx() {
  local f
  # 1. conda recipes: drop the ucx dependency (host/run/ignore_run_exports).
  for f in conda/recipes/libucxx/recipe.yaml conda/recipes/ucxx/recipe.yaml; do
    sed -i -E '/^\s*- ucx(\s*|\s+[<>=].*)$/d' "$f"
    grep -qE '^\s*- ucx(\s|$)' "$f" \
      && { echo "ERROR: ucx dep strip did not apply to $f" >&2; return 1; }
    # linking against a lib outside the recipe deps is now intended
    sed -i 's#overlinking_behavior: "error"#overlinking_behavior: "ignore"#' "$f"
    # the builds find the PR's UCX via CMake, and the build-time imports of
    # the fresh python module load its libraries via LD_LIBRARY_PATH
    grep -q "CMAKE_PREFIX_PATH=$UCX_PR_PREFIX" "$f" \
      || sed -i -E "s#^([[:space:]]*)(\./build\.sh .*)#\1export CMAKE_PREFIX_PATH=$UCX_PR_PREFIX\n\1export LD_LIBRARY_PATH=$UCX_PR_PREFIX/lib\\\${LD_LIBRARY_PATH:+:\\\$LD_LIBRARY_PATH}\n\1\2#" "$f"
    grep -q "CMAKE_PREFIX_PATH=$UCX_PR_PREFIX" "$f" \
      || { echo "ERROR: CMAKE_PREFIX_PATH inject did not apply to $f" >&2; return 1; }
  done
  # 2. dependencies.yaml: drop the released ucx/libucx package deps
  #    (feeds the wheel pyproject deps and the conda test/docs envs).
  sed -i -E '/^\s*- (lib)?ucx[<>=]/d; /^\s*- libucx-cu1[23][<>=]/d' dependencies.yaml
  grep -qE '^\s*- (lib)?ucx[<>=]|^\s*- libucx-cu1[23][<>=]' dependencies.yaml \
    && { echo "ERROR: ucx dep strip did not apply to dependencies.yaml" >&2; return 1; }
  # 3. wheel builds: CMake finds the PR's UCX; auditwheel resolves libucp
  #    from it at repair time (libucp.so.0 stays excluded from the wheel).
  # CMAKE_PREFIX_PATH via the environment: additive - CMake consults it on
  # top of the prefix paths scikit-build-core injects (site-packages, where
  # the libucxx wheel's cmake config lives).
  for f in ci/build_wheel_libucxx.sh ci/build_wheel_ucxx.sh; do
    grep -q "ucx-pr" "$f" && continue
    sed -i "s#^export SKBUILD_CMAKE_ARGS=\"#export CMAKE_PREFIX_PATH=$UCX_PR_PREFIX\${CMAKE_PREFIX_PATH:+:\$CMAKE_PREFIX_PATH}\nexport LD_LIBRARY_PATH=$UCX_PR_PREFIX/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\nexport SKBUILD_CMAKE_ARGS=\"#" "$f"
    grep -q "ucx-pr" "$f" \
      || { echo "ERROR: UCX-PR wheel patch did not apply to $f" >&2; return 1; }
  done
  # 4. The recipe package self-tests import ucxx in a fresh ephemeral env that
  #    has no ucx (by design now) - skip them; the pipeline's test jobs are
  #    the real coverage.
  for f in ci/build_cpp.sh ci/build_python.sh; do
    grep -q "test skip" "$f" \
      || sed -i "s#rattler-build build#rattler-build build --test skip#" "$f"
    grep -q "test skip" "$f" \
      || { echo "ERROR: rattler test-skip patch did not apply to $f" >&2; return 1; }
  done
  return 0
}
