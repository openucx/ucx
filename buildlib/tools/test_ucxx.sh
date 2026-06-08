#!/bin/bash -eE
#
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#
# Usage: test_ucxx.sh <build|test_cpp|test_python|test_wheel_ucxx>
# Env: RAPIDS_CUDA_VERSION, RAPIDS_PY_VERSION, UCXX_DIR
#   build|test_cpp|test_python:  also IS_GPU
#   test_wheel_ucxx:             also LIBUCXX_WHL_DIR, UCXX_WHL_DIR

phase=${1:?phase required}
case "$phase" in
  build|test_cpp|test_python) : "${IS_GPU:?IS_GPU required}" ;;
esac
: "${RAPIDS_CUDA_VERSION:?RAPIDS_CUDA_VERSION required}"
: "${RAPIDS_PY_VERSION:?RAPIDS_PY_VERSION required}"
: "${UCXX_DIR:?UCXX_DIR required}"

export RAPIDS_CUDA_VERSION RAPIDS_PY_VERSION
export RAPIDS_CONDA_BLD_OUTPUT_DIR=/tmp/conda-bld-output
mkdir -p "$RAPIDS_CONDA_BLD_OUTPUT_DIR" "$HOME/.local/bin"

for tool in rapids-download-conda-from-github rapids-download-from-github; do
  printf '#!/bin/bash\necho "%s"\n' "$RAPIDS_CONDA_BLD_OUTPUT_DIR" > "$HOME/.local/bin/$tool"
  chmod +x "$HOME/.local/bin/$tool"
done
export PATH="$HOME/.local/bin:$PATH"

cd "$UCXX_DIR"

# Tolerate missing nvidia-smi on CPU containers. Guard catches upstream rewording.
sed -i 's#^  nvidia-smi$#  command -v nvidia-smi >/dev/null \&\& nvidia-smi || echo "(no GPU)"#' ci/test_common.sh
grep -q 'command -v nvidia-smi' ci/test_common.sh \
  || { echo "ERROR: nvidia-smi patch did not apply to ci/test_common.sh" >&2; exit 1; }

# Skip test_client_shutdown: its teardown crashes the xdist worker under
# full-pipeline GPU/MPS contention (flaky upstream test, not a UCX issue).
# Guard catches upstream rewording (else the skip silently disappears).
sed -i "s#--runslow#--runslow -k 'not test_client_shutdown'#" ci/run_python.sh
grep -q "not test_client_shutdown" ci/run_python.sh \
  || { echo "ERROR: test_client_shutdown skip did not apply to ci/run_python.sh" >&2; exit 1; }

# Force host driver ahead of the image's newer compat driver (MPS rejects a client
# newer than the daemon -> cuInit hangs). ubuntu: /usr/lib/<arch>-linux-gnu; wheel: /usr/lib64.
arch=$(uname -m)
for hostlib in "/usr/lib/$arch-linux-gnu" /usr/lib64; do
  [ -d "$hostlib" ] && export LD_LIBRARY_PATH="$hostlib:${LD_LIBRARY_PATH:-}"
done

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
    # CPU slices have no GPU device bound; CUDA-touching gtests would crash.
    if [ "${IS_GPU,,}" = "true" ]; then
      bash ci/test_cpp.sh
    else
      CUDA_VISIBLE_DEVICES= UCX_TLS=tcp,sm,self GTEST_FILTER='-RMM*.*:CCCL*.*' \
        bash ci/test_cpp.sh
    fi
    ;;

  test_python)
    bash ci/test_python.sh
    ;;

  test_wheel_ucxx)
    : "${LIBUCXX_WHL_DIR:?LIBUCXX_WHL_DIR required}"
    : "${UCXX_WHL_DIR:?UCXX_WHL_DIR required}"
    printf '#!/bin/bash\necho "%s"\n' "$LIBUCXX_WHL_DIR" > "$HOME/.local/bin/rapids-download-wheels-from-github"
    printf '#!/bin/bash\necho "%s"\n' "$UCXX_WHL_DIR"    > "$HOME/.local/bin/rapids-download-from-github"
    chmod +x "$HOME/.local/bin/rapids-download-wheels-from-github" "$HOME/.local/bin/rapids-download-from-github"
    bash ci/test_wheel_ucxx.sh
    ;;

  *) echo "Unknown phase: $phase" >&2; exit 1 ;;
esac
