#!/bin/bash -eE
#
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#
# Usage: test_ucxx.sh <build|test_cpp|test_python|test_wheel_ucxx>
# Env: RAPIDS_CUDA_VERSION, RAPIDS_PY_VERSION, UCXX_DIR
#   build|test_cpp|test_python:  also IS_GPU
#   test_wheel_ucxx:             also LIBUCXX_WHL_DIR, UCXX_WHL_DIR
#
# Everything runs against the UCX built from this checkout (the PR under
# test): the released ucx/libucx dependencies are stripped from the ucxx tree
# (see ucxx_ucx_pr.sh), the packages build against UCX_PR_PREFIX, and the
# tests load its libraries.

# The pipeline runs this as `bash <path>`, which ignores the shebang options,
# so the shell flags have to be set here for the fail-loud guards to bite.
set -eE -o pipefail

phase=${1:?phase required}
case "$phase" in
  build|test_cpp|test_python) : "${IS_GPU:?IS_GPU required}" ;;
esac
: "${RAPIDS_CUDA_VERSION:?RAPIDS_CUDA_VERSION required}"
: "${RAPIDS_PY_VERSION:?RAPIDS_PY_VERSION required}"
: "${UCXX_DIR:?UCXX_DIR required}"

ucx_dir=$(cd "$(dirname "$0")/../.." && pwd)
export ucx_dir UCX_PR_PREFIX=/tmp/ucx-pr
source "$ucx_dir/buildlib/tools/ucxx_ucx_pr.sh"

# The PR's UCX was built once by the UCXX_ucx stage and staged at
# UCX_PR_PREFIX by the fetch step; every build and test consumes it.
[ -x "$UCX_PR_PREFIX/bin/ucx_info" ] \
  || { echo "ERROR: UCX (PR) not staged at $UCX_PR_PREFIX - UCXX_ucx artifact missing?" >&2; exit 1; }

export RAPIDS_CUDA_VERSION RAPIDS_PY_VERSION
# upstream test-env matrices select the dependency set via this key
export RAPIDS_DEPENDENCIES=${RAPIDS_DEPENDENCIES:-latest}
# persistent pip cache (host volume); pip disables caching itself if unwritable
[ -d /pip-cache ] && export PIP_CACHE_DIR=/pip-cache
export RAPIDS_CONDA_BLD_OUTPUT_DIR=/tmp/conda-bld-output
mkdir -p "$RAPIDS_CONDA_BLD_OUTPUT_DIR" "$HOME/.local/bin"

# The ucxx test scripts fetch the packages built by the previous step through
# rapids-download-from-github; shim it to the locally built channel.
printf '#!/bin/bash\necho "%s"\n' "$RAPIDS_CONDA_BLD_OUTPUT_DIR" \
  > "$HOME/.local/bin/rapids-download-from-github"
chmod +x "$HOME/.local/bin/rapids-download-from-github"
export PATH="$HOME/.local/bin:$PATH"

cd "$UCXX_DIR"

# Point every ucxx build and test at the PR's UCX instead of released ucx.
use_pr_ucx

# Tolerate missing nvidia-smi on CPU containers. Guard catches upstream rewording.
sed -i 's#^  nvidia-smi$#  command -v nvidia-smi >/dev/null \&\& nvidia-smi || echo "(no GPU)"#' ci/test_common.sh
grep -q 'command -v nvidia-smi' ci/test_common.sh \
  || { echo "ERROR: nvidia-smi patch did not apply to ci/test_common.sh" >&2; exit 1; }

# Conda tests load the PR-built UCX: right after env activation, copy its
# libraries and tools into the env (the env carries no ucx package - the PR
# build is the only provider) and verify the lib the loader sees is the PR's.
for f in ci/test_cpp.sh ci/test_python.sh; do
  grep -q "ucx-pr" "$f" \
    || sed -i 's#^conda activate test$#conda activate test\ncp -a /tmp/ucx-pr/lib/. "$CONDA_PREFIX/lib/"\ncp -a /tmp/ucx-pr/bin/. "$CONDA_PREFIX/bin/"\ncmp -s "$CONDA_PREFIX/lib/libucs.so.0" /tmp/ucx-pr/lib/libucs.so.0 || { echo "ERROR: UCX-PR overlay verification failed" >\&2; exit 1; }\necho "UCX-PR overlaid into test env"#' "$f"
  grep -q "ucx-pr" "$f" \
    || { echo "ERROR: UCX-PR overlay patch did not apply to $f" >&2; exit 1; }
done

# Forces the host driver ahead of the image's newer compat driver (MPS rejects
# a client newer than the daemon -> cuInit hangs). ubuntu: /usr/lib/<arch>-linux-gnu;
# wheel: /usr/lib64. Test-runtime only: with this in LD_LIBRARY_PATH, ld resolves
# shared-lib dependencies against the image's newer glibc and the UCX build fails,
# so the build steps run without it.
export_host_driver_override() {
  local arch hostlib
  arch=$(uname -m)
  for hostlib in "/usr/lib/$arch-linux-gnu" /usr/lib64; do
    [ -d "$hostlib" ] && export LD_LIBRARY_PATH="$hostlib:${LD_LIBRARY_PATH:-}"
  done
  # arm64 images carry no /usr/lib64, so the loop ends non-zero there; the
  # caller runs under errexit and must not treat that as a failure.
  return 0
}

case "$phase" in
  build)
    if [ "${IS_GPU,,}" = "true" ]; then
      # sccache wrapper crashes CMake's compiler probe on the GPU build hosts; no-op it.
      cat > "$HOME/.local/bin/rapids-configure-sccache" <<'EOF'
#!/bin/bash
export CMAKE_C_COMPILER_LAUNCHER= CMAKE_CXX_COMPILER_LAUNCHER= CMAKE_CUDA_COMPILER_LAUNCHER= RUSTC_WRAPPER=
EOF
      chmod +x "$HOME/.local/bin/rapids-configure-sccache"
    fi
    bash ci/build_cpp.sh
    bash ci/build_python.sh
    ;;

  test_cpp)
    export_host_driver_override
    # CPU slices have no GPU device bound; CUDA-touching gtests would crash.
    if [ "${IS_GPU,,}" = "true" ]; then
      bash ci/test_cpp.sh
    else
      CUDA_VISIBLE_DEVICES= UCX_TLS=tcp,sm,self GTEST_FILTER='-RMM*.*:CCCL*.*' \
        bash ci/test_cpp.sh
    fi
    ;;

  test_python)
    export_host_driver_override
    bash ci/test_python.sh
    ;;

  test_wheel_ucxx)
    : "${LIBUCXX_WHL_DIR:?LIBUCXX_WHL_DIR required}"
    : "${UCXX_WHL_DIR:?UCXX_WHL_DIR required}"
    # The ucxx wheel carries no libucx dependency - the loader resolves the
    # UCX libraries from UCX_PR_PREFIX via LD_LIBRARY_PATH, and the injected
    # check asserts the runtime UCX version is the PR's.
    grep -q "ucx-pr" ci/test_wheel_ucxx.sh \
      || sed -i '/^print_system_stats$/i want=$(/tmp/ucx-pr/bin/ucx_info -v | sed -n "s/^# Library version: //p")\ngot=$(python -c "import ucxx; print(*ucxx.get_ucx_version(), sep=chr(46))")\n[ "$got" = "$want" ] || { echo "ERROR: UCX version mismatch: $got != $want" >\&2; exit 1; }\necho "UCXX runs UCX-PR $got"' ci/test_wheel_ucxx.sh
    grep -q "ucx-pr" ci/test_wheel_ucxx.sh \
      || { echo "ERROR: UCX-PR version check did not apply to ci/test_wheel_ucxx.sh" >&2; exit 1; }
    # The wheel test installs from both the libucxx and ucxx wheelhouses via
    # the download helpers; stage both wheels in one dir and point the helpers
    # there so the libucxx_*.whl / ucxx_*.whl globs resolve.
    wheelhouse="$RAPIDS_CONDA_BLD_OUTPUT_DIR/ucxx-wheelhouse"
    mkdir -p "$wheelhouse"
    cp "$LIBUCXX_WHL_DIR"/*.whl "$UCXX_WHL_DIR"/*.whl "$wheelhouse"/
    printf '#!/bin/bash\necho "%s"\n' "$wheelhouse" \
      > "$HOME/.local/bin/rapids-download-from-github"
    chmod +x "$HOME/.local/bin/rapids-download-from-github"
    export_host_driver_override
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$UCX_PR_PREFIX/lib"
    bash ci/test_wheel_ucxx.sh
    ;;

  *) echo "Unknown phase: $phase" >&2; exit 1 ;;
esac
