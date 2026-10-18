#!/bin/bash -eE
#
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#
# Usage: test_ucxx.sh <test_cpp|test_python|test_wheel_ucxx>
# Env: RAPIDS_CUDA_VERSION, RAPIDS_PY_VERSION, UCXX_DIR
#   test_cpp|test_python:  also IS_GPU, CONDA_CHANNEL_DIR (the conda-python
#                          build artifact: a channel with both the libucxx and
#                          the ucxx packages)
#   test_wheel_ucxx:       also LIBUCXX_WHL_DIR, UCXX_WHL_DIR
#
# Everything runs against the UCX built from this checkout (the PR under
# test): the ucxx UCXX_NO_LIBUCX switch drops the released ucx/libucx
# dependencies, the packages under test were built against UCX_PR_PREFIX by
# the UCXX_build jobs, and the tests load its libraries.

# The pipeline runs this as `bash <path>`, which ignores the shebang options,
# so the shell flags have to be set here for the fail-loud guards to bite.
set -eE -o pipefail

# RAPIDS CI scripts require it whenever CI is set.
export RAPIDS_DATETIME_STRING=${RAPIDS_DATETIME_STRING:-$(date -u +%y%m%d%H%M%S)}

phase=${1:?phase required}
case "$phase" in
  test_cpp|test_python)
    : "${IS_GPU:?IS_GPU required}"
    : "${CONDA_CHANNEL_DIR:?CONDA_CHANNEL_DIR required}" ;;
esac
: "${RAPIDS_CUDA_VERSION:?RAPIDS_CUDA_VERSION required}"
: "${RAPIDS_PY_VERSION:?RAPIDS_PY_VERSION required}"
: "${UCXX_DIR:?UCXX_DIR required}"

UCX_PR_PREFIX=/tmp/ucx-pr

# The PR's UCX was built once by the UCXX_ucx stage and staged at
# UCX_PR_PREFIX by the fetch step; every build and test consumes it.
[ -x "$UCX_PR_PREFIX/bin/ucx_info" ] \
  || { echo "ERROR: UCX (PR) not staged at $UCX_PR_PREFIX - UCXX_ucx artifact missing?" >&2; exit 1; }

export RAPIDS_CUDA_VERSION RAPIDS_PY_VERSION
# the ucxx test-env matrices select the dependency set via this key
export RAPIDS_DEPENDENCIES=${RAPIDS_DEPENDENCIES:-latest}
# persistent pip cache (host volume); pip disables caching itself if unwritable
[ -d /pip-cache ] && export PIP_CACHE_DIR=/pip-cache
mkdir -p "$HOME/.local/bin"
export PATH="$HOME/.local/bin:$PATH"

# The ucxx test scripts fetch the packages of the build jobs through
# rapids-download-from-github; shim it to the staged artifact dir.
shim_download_helper() {
  printf '#!/bin/bash\necho "%s"\n' "$1" > "$HOME/.local/bin/rapids-download-from-github"
  chmod +x "$HOME/.local/bin/rapids-download-from-github"
}

cd "$UCXX_DIR"

# Point every ucxx test at the PR's UCX instead of released ucx: with the
# ucxx UCXX_NO_LIBUCX switch the test scripts take UCX from ucx_ROOT, and the
# wheel test checks that the UCX it loaded is that one.
export UCXX_NO_LIBUCX=true ucx_ROOT="$UCX_PR_PREFIX"

# Forces the host driver ahead of the image's newer compat driver (MPS rejects
# a client newer than the daemon -> cuInit hangs). ubuntu: /usr/lib/<arch>-linux-gnu;
# wheel: /usr/lib64. Test-runtime only: with this in LD_LIBRARY_PATH, ld resolves
# shared-lib dependencies against the image's newer glibc and the UCX build fails,
# so the build steps run without it.
export_host_driver_override() {
  local arch hostlib
  arch=$(uname -m)
  for hostlib in "/usr/lib/$arch-linux-gnu" /usr/lib64; do
    [ -d "$hostlib" ] && export LD_LIBRARY_PATH="$hostlib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  done
  # arm64 images carry no /usr/lib64, so the loop ends non-zero there; the
  # caller runs under errexit and must not treat that as a failure.
  return 0
}

case "$phase" in
  test_cpp)
    shim_download_helper "$CONDA_CHANNEL_DIR"
    export_host_driver_override
    # CPU slices have no GPU device bound; CUDA-touching gtests would crash.
    if [ "${IS_GPU,,}" = "true" ]; then
      bash ci/test_cpp.sh
    else
      CUDA_VISIBLE_DEVICES= UCX_TLS=tcp,sm,self GTEST_FILTER='-RMM*.*:CCCL*.*:ListenerCuda*.*' \
        bash ci/test_cpp.sh
    fi
    ;;

  test_python)
    shim_download_helper "$CONDA_CHANNEL_DIR"
    export_host_driver_override
    bash ci/test_python.sh
    ;;

  test_wheel_ucxx)
    : "${LIBUCXX_WHL_DIR:?LIBUCXX_WHL_DIR required}"
    : "${UCXX_WHL_DIR:?UCXX_WHL_DIR required}"
    # The ucxx wheel carries no libucx dependency - the loader resolves the
    # UCX libraries from UCX_PR_PREFIX via LD_LIBRARY_PATH.
    # The wheel test installs from both the libucxx and ucxx wheelhouses via
    # the download helpers; stage both wheels in one dir and point the helpers
    # there so the libucxx_*.whl / ucxx_*.whl globs resolve.
    wheelhouse=/tmp/ucxx-wheelhouse
    mkdir -p "$wheelhouse"
    cp "$LIBUCXX_WHL_DIR"/*.whl "$UCXX_WHL_DIR"/*.whl "$wheelhouse"/
    shim_download_helper "$wheelhouse"
    export_host_driver_override
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$UCX_PR_PREFIX/lib"
    bash ci/test_wheel_ucxx.sh
    ;;

  *) echo "Unknown phase: $phase" >&2; exit 1 ;;
esac
