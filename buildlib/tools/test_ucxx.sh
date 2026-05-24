#!/bin/bash -eE
#
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See file LICENSE for terms.
#
# UCXX_tests stage runner.
# Usage: test_ucxx.sh <build|test_cpp|test_python>
# Env: IS_GPU, RAPIDS_CUDA_VERSION, RAPIDS_PY_VERSION, UCXX_DIR

phase=${1:?phase required: build | test_cpp | test_python}
: "${IS_GPU:?IS_GPU required}"
: "${RAPIDS_CUDA_VERSION:?RAPIDS_CUDA_VERSION required}"
: "${RAPIDS_PY_VERSION:?RAPIDS_PY_VERSION required}"
: "${UCXX_DIR:?UCXX_DIR required}"

export RAPIDS_CUDA_VERSION RAPIDS_PY_VERSION
export RAPIDS_CONDA_BLD_OUTPUT_DIR=/tmp/conda-bld-output
mkdir -p "$RAPIDS_CONDA_BLD_OUTPUT_DIR"

# Shim RAPIDS gha-tools downloaders to local conda-bld dir.
mkdir -p "$HOME/.local/bin"
for tool in rapids-download-conda-from-github rapids-download-from-github; do
  printf '#!/bin/bash\necho "%s"\n' "$RAPIDS_CONDA_BLD_OUTPUT_DIR" > "$HOME/.local/bin/$tool"
  chmod +x "$HOME/.local/bin/$tool"
done
export PATH="$HOME/.local/bin:$PATH"

cd "$UCXX_DIR"

case "$phase" in
  build)
    echo "=== id: $(id), arch: $(uname -m), gpu: $IS_GPU ==="
    which python && python --version
    [ "$IS_GPU" = "true" ] && nvidia-smi || true

    if [ "$IS_GPU" = "true" ]; then
      # sccache from rapids-configure-sccache crashes on CMake TryCompile in
      # this image; replace with a no-op that still exports the env vars the
      # rattler recipe expects.
      cat > "$HOME/.local/bin/rapids-configure-sccache" <<'EOF'
#!/bin/bash
export CMAKE_C_COMPILER_LAUNCHER=
export CMAKE_CXX_COMPILER_LAUNCHER=
export CMAKE_CUDA_COMPILER_LAUNCHER=
export RUSTC_WRAPPER=
EOF
      chmod +x "$HOME/.local/bin/rapids-configure-sccache"

      # Upstream header missing <unistd.h>.
      hdr=python/ucxx/ucxx/examples/python_future_task.h
      grep -q "include <unistd.h>" "$hdr" || sed -i '/^#pragma once/a #include <unistd.h>' "$hdr"
    else
      # No GPU on CPU slice; soften nvidia-smi check.
      sed -i 's#^  nvidia-smi$#  command -v nvidia-smi >/dev/null \&\& nvidia-smi || echo "(no GPU - CPU slice)"#' ci/test_common.sh
    fi

    bash ci/build_cpp.sh
    bash ci/build_python.sh
    ;;

  test_cpp)
    if [ "$IS_GPU" = "true" ]; then
      # Run as Azure-injected step user (mapped to host swx-azure-svc uid),
      # which matches the host MPS daemon owner so MPS accepts the client.
      bash ci/test_cpp.sh
    else
      # CPU slice: no GPU driver loaded. Pin UCX onto host transports;
      # filter CUDA-only gtest suites.
      export CUDA_VISIBLE_DEVICES=
      export UCX_TLS=tcp,sm,self
      export GTEST_FILTER='-RMM*.*:CCCL*.*'
      bash ci/test_cpp.sh
    fi
    ;;

  test_python)
    if [ "$IS_GPU" != "true" ]; then
      echo "test_python only runs on GPU slice; skipping" >&2
      exit 0
    fi
    bash ci/test_python.sh
    ;;

  *)
    echo "Unknown phase: $phase (expected build | test_cpp | test_python)" >&2
    exit 1
    ;;
esac
