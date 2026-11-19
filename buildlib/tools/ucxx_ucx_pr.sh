# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#
# Sourced by build_ucxx.sh. Provides:
#   build_ucx_pr_conda  - build the PR's UCX into UCX_PR_PREFIX via a
#                         conda-forge toolchain env
# Requires: ucx_dir, UCX_PR_PREFIX exported by the caller.

# This file's own path, so build_ucx_pr_conda can re-source it in the toolchain
# shell instead of relying on an exported function surviving conda run.
ucxx_ucx_pr_sh=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")

# Runs inside the toolchain env created by build_ucx_pr_conda, which sources
# this file to get it. Idempotent - the container, and thus /tmp, is shared
# by all steps of a job.
build_ucx_pr() {
  [ -x "$UCX_PR_PREFIX/bin/ucx_info" ] && return 0
  # CUDA comes from the env's conda-forge cuda dev packages, pinned to
  # RAPIDS_CUDA_VERSION; headers and stubs live under targets/<triplet>-linux,
  # where the triplet is x86_64 on x86 and sbsa on aarch64.
  local cuda_h cuda_dir
  cuda_h=$(ls "${CONDA_PREFIX:?CONDA_PREFIX required}"/targets/*-linux/include/cuda.h 2>&1) \
    || { echo "ERROR: no cuda.h under $CONDA_PREFIX/targets/*-linux: $cuda_h" >&2; return 1; }
  cuda_dir=${cuda_h%/include/cuda.h}
  echo "UCX-PR configure cuda: --with-cuda=$cuda_dir"
  (cd "$ucx_dir" \
   && ./autogen.sh \
   && ./contrib/configure-release --prefix="$UCX_PR_PREFIX" --with-cuda="$cuda_dir" \
        --enable-mt --without-java --without-go --disable-doxygen-doc \
   && make -j"$(nproc)" install) > /tmp/ucx-pr-build.log 2>&1 \
    || { tail -50 /tmp/ucx-pr-build.log >&2; echo "ERROR: UCX (PR) build failed" >&2; return 1; }
}

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
  conda run -n ucx-build bash -ec "source '$ucxx_ucx_pr_sh'; build_ucx_pr" \
    || { echo "ERROR: UCX (PR) build failed in toolchain env" >&2; return 1; }
}
