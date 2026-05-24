#!/bin/bash -eE
#
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See file LICENSE for terms.
#
# UCXX_build stage runner.
# Usage: build_ucxx.sh <conda_cpp|conda_python|wheel_libucxx|wheel_ucxx|wheel_distributed_ucxx>
# Env: RAPIDS_CUDA_VERSION, RAPIDS_PY_VERSION, UCXX_DIR, RAPIDS_BLD_OUTPUT_DIR

phase=${1:?phase required}
: "${RAPIDS_CUDA_VERSION:?RAPIDS_CUDA_VERSION required}"
: "${RAPIDS_PY_VERSION:?RAPIDS_PY_VERSION required}"
: "${UCXX_DIR:?UCXX_DIR required}"
: "${RAPIDS_BLD_OUTPUT_DIR:?RAPIDS_BLD_OUTPUT_DIR required}"

export RAPIDS_CUDA_VERSION RAPIDS_PY_VERSION
mkdir -p "$RAPIDS_BLD_OUTPUT_DIR"

# Upstream scripts read phase-specific output-dir env names.
case "$phase" in
  conda_*)  export RAPIDS_CONDA_BLD_OUTPUT_DIR="$RAPIDS_BLD_OUTPUT_DIR" ;;
  wheel_*)  export RAPIDS_WHEEL_BLD_OUTPUT_DIR="$RAPIDS_BLD_OUTPUT_DIR" ;;
esac

# Shim RAPIDS gha-tools downloaders to local build output dir.
mkdir -p "$HOME/.local/bin"
for tool in rapids-download-conda-from-github rapids-download-from-github; do
  printf '#!/bin/bash\necho "%s"\n' "$RAPIDS_BLD_OUTPUT_DIR" > "$HOME/.local/bin/$tool"
  chmod +x "$HOME/.local/bin/$tool"
done

# sccache from rapids-configure-sccache crashes on CMake TryCompile in this
# image; replace with a no-op that still exports the env vars rattler expects.
cat > "$HOME/.local/bin/rapids-configure-sccache" <<'EOF'
#!/bin/bash
export CMAKE_C_COMPILER_LAUNCHER=
export CMAKE_CXX_COMPILER_LAUNCHER=
export CMAKE_CUDA_COMPILER_LAUNCHER=
export RUSTC_WRAPPER=
EOF
chmod +x "$HOME/.local/bin/rapids-configure-sccache"

# wheel_ucxx phase consumes a libucxx wheel artifact; shim downloader to it.
if [ -n "${WHEEL_INPUT_DIR:-}" ]; then
  printf '#!/bin/bash\necho "%s"\n' "$WHEEL_INPUT_DIR" > "$HOME/.local/bin/rapids-download-wheels-from-github"
  chmod +x "$HOME/.local/bin/rapids-download-wheels-from-github"
fi

export PATH="$HOME/.local/bin:$PATH"

cd "$UCXX_DIR"

# Upstream header missing <unistd.h>.
hdr=python/ucxx/ucxx/examples/python_future_task.h
grep -q "include <unistd.h>" "$hdr" || sed -i '/^#pragma once/a #include <unistd.h>' "$hdr"

echo "=== id: $(id), arch: $(uname -m), phase: $phase ==="

# Wheel image (Rocky 8) defaults to gcc 8.5; UCXX C++ needs gcc-toolset-14
# (designated initializers in assignment, matches upstream wheel CI).
case "$phase" in
  wheel_*)
    toolset=/opt/rh/gcc-toolset-14/root/usr/bin
    if [ -d "$toolset" ]; then
      export PATH="$toolset:$PATH"
      export CC="$toolset/gcc"
      export CXX="$toolset/g++"
    fi
    ;;
esac

case "$phase" in
  conda_cpp)             bash ci/build_cpp.sh ;;
  conda_python)          bash ci/build_python.sh ;;
  wheel_libucxx)         bash ci/build_wheel_libucxx.sh ;;
  wheel_ucxx)            bash ci/build_wheel_ucxx.sh ;;
  wheel_distributed_ucxx) bash ci/build_wheel_distributed_ucxx.sh ;;
  *) echo "Unknown phase: $phase" >&2; exit 1 ;;
esac
