#!/bin/bash -eE
#
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See file LICENSE for terms.
#
# Usage: build_ucxx.sh <conda_cpp|conda_python|wheel_libucxx|wheel_ucxx|docs|devcontainer>
# Env: UCXX_DIR (all phases). Build phases also need RAPIDS_CUDA_VERSION,
#   RAPIDS_PY_VERSION, RAPIDS_BLD_OUTPUT_DIR.
# wheel_ucxx phase also requires WHEEL_INPUT_DIR (libucxx wheel artifact dir)
# Docs phase env: CPP_CHANNEL_DIR, PYTHON_CHANNEL_DIR, RAPIDS_DOCS_DIR

phase=${1:?phase required}
: "${UCXX_DIR:?UCXX_DIR required}"

case "$phase" in
  devcontainer)
    # Parse each .devcontainer config; verify its Dockerfile + BASE exist
    # (no registry pull - the devcontainer CLI catches missing images at use).
    UCXX_DIR="$UCXX_DIR" python3 - <<'PY'
import glob, json, os, sys
root = os.environ["UCXX_DIR"]
cfgs = glob.glob(os.path.join(root, ".devcontainer", "*", "devcontainer.json"))
if not cfgs:
    sys.exit("ERROR: no devcontainer.json under .devcontainer/")
for cfg in cfgs:
    b = json.load(open(cfg))["build"]
    df = b["dockerfile"].replace("${localWorkspaceFolder}", root)
    assert os.path.isfile(df), f"{cfg}: missing Dockerfile {df}"
    assert b["args"]["BASE"], f"{cfg}: empty BASE"
    print(f"OK {cfg}")
PY
    exit 0 ;;
esac

: "${RAPIDS_CUDA_VERSION:?RAPIDS_CUDA_VERSION required}"
: "${RAPIDS_PY_VERSION:?RAPIDS_PY_VERSION required}"
: "${RAPIDS_BLD_OUTPUT_DIR:?RAPIDS_BLD_OUTPUT_DIR required}"

export RAPIDS_CUDA_VERSION RAPIDS_PY_VERSION
mkdir -p "$RAPIDS_BLD_OUTPUT_DIR"

case "$phase" in
  conda_*) export RAPIDS_CONDA_BLD_OUTPUT_DIR="$RAPIDS_BLD_OUTPUT_DIR" ;;
  wheel_*) export RAPIDS_WHEEL_BLD_OUTPUT_DIR="$RAPIDS_BLD_OUTPUT_DIR" ;;
  docs)
    : "${CPP_CHANNEL_DIR:?CPP_CHANNEL_DIR required for docs phase}"
    : "${PYTHON_CHANNEL_DIR:?PYTHON_CHANNEL_DIR required for docs phase}"
    : "${RAPIDS_DOCS_DIR:?RAPIDS_DOCS_DIR required for docs phase}"
    mkdir -p "$RAPIDS_DOCS_DIR" ;;
esac

mkdir -p "$HOME/.local/bin"
for tool in rapids-download-conda-from-github rapids-download-from-github; do
  printf '#!/bin/bash\necho "%s"\n' "$RAPIDS_BLD_OUTPUT_DIR" > "$HOME/.local/bin/$tool"
  chmod +x "$HOME/.local/bin/$tool"
done
# Docs phase: override shims to point at the staged conda channels.
if [ "$phase" = "docs" ]; then
  printf '#!/bin/bash\necho "%s"\n' "$CPP_CHANNEL_DIR"    > "$HOME/.local/bin/rapids-download-conda-from-github"
  printf '#!/bin/bash\necho "%s"\n' "$PYTHON_CHANNEL_DIR" > "$HOME/.local/bin/rapids-download-from-github"
fi

# Point the wheel-download helpers at the staged libucxx wheelhouse so the
# wheel_ucxx build resolves it.
if [ -n "${WHEEL_INPUT_DIR:-}" ]; then
  for tool in rapids-download-from-github rapids-download-wheels-from-github; do
    printf '#!/bin/bash\necho "%s"\n' "$WHEEL_INPUT_DIR" > "$HOME/.local/bin/$tool"
    chmod +x "$HOME/.local/bin/$tool"
  done
fi

export PATH="$HOME/.local/bin:$PATH"

cd "$UCXX_DIR"

# Wheel builds otherwise pick system gcc 8.5 (too old for libucxx's C++20);
# point CC/CXX at gcc-toolset-14.
if [[ "$phase" == wheel_* ]]; then
  toolset=/opt/rh/gcc-toolset-14/root/usr/bin
  [ -x "$toolset/gcc" ] \
    || { echo "ERROR: gcc-toolset-14 not found at $toolset (needed for libucxx C++20)" >&2; exit 1; }
  export CC="$toolset/gcc" CXX="$toolset/g++"
fi

case "$phase" in
  conda_cpp)              bash ci/build_cpp.sh ;;
  conda_python)           bash ci/build_python.sh ;;
  wheel_libucxx)          bash ci/build_wheel_libucxx.sh ;;
  wheel_ucxx)
    : "${WHEEL_INPUT_DIR:?WHEEL_INPUT_DIR required for wheel_ucxx (libucxx wheel dir)}"
    bash ci/build_wheel_ucxx.sh ;;
  docs)
    # Upstream forces RAPIDS_DOCS_DIR=$(mktemp -d); make it default-if-unset
    # so our staged output dir survives. Guard catches upstream rewording.
    sed -i 's|RAPIDS_DOCS_DIR="$(mktemp -d)"|: "${RAPIDS_DOCS_DIR:=$(mktemp -d)}"|' ci/build_docs.sh
    grep -q 'RAPIDS_DOCS_DIR:=' ci/build_docs.sh \
      || { echo "ERROR: docs patch did not apply to ci/build_docs.sh" >&2; exit 1; }
    bash ci/build_docs.sh ;;
  *) echo "Unknown phase: $phase" >&2; exit 1 ;;
esac
