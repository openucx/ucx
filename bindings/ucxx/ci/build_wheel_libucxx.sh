#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

source rapids-init-pip

package_name="libucxx"
package_dir="python/libucxx"

rapids-logger "Generating build requirements"

rapids-dependency-file-generator \
  --output requirements \
  --file-key "py_build_${package_name}" \
  --file-key "py_rapids_build_${package_name}" \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};cuda_suffixed=true;no_libucx=${UCXX_NO_LIBUCX:-false}" \
| tee /tmp/requirements-build.txt

rapids-logger "Installing build requirements"
rapids-pip-retry install \
    -v \
    --prefer-binary \
    -r /tmp/requirements-build.txt

# build with '--no-build-isolation', for better sccache hit rate
# 0 really means "add --no-build-isolation" (ref: https://github.com/pypa/pip/issues/5735)
export PIP_NO_BUILD_ISOLATION=0

export SKBUILD_CMAKE_ARGS="-DUCXX_ENABLE_CCCL=ON"

# Propagate 'no_libucx' to rapids-build-backend, which regenerates the wheel's
# dependencies from dependencies.yaml at build time. This overrides (does not extend)
# 'matrix-entry' from pyproject.toml, so the existing entries must be repeated here.
# See ci/README.md for UCXX_NO_LIBUCX requirements and limitations.
export RAPIDS_MATRIX_ENTRY="cuda_suffixed=true;no_libucx=${UCXX_NO_LIBUCX:-false}"

./ci/build_wheel.sh "${package_name}" "${package_dir}"

python -m auditwheel repair \
    --exclude "libucp.so.0" \
    -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
    ${package_dir}/dist/*

./ci/validate_wheel.sh "${package_dir}" "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

RAPIDS_PACKAGE_NAME="$(rapids-artifact-name wheel_cpp libucxx ucxx --cuda "${RAPIDS_CUDA_VERSION}")"
export RAPIDS_PACKAGE_NAME
