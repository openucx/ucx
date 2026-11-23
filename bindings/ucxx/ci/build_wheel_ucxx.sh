#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

source rapids-init-pip

package_name="ucxx"
package_dir="python/ucxx"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Downloads libucxx wheel from this current build,
# then ensures 'ucxx' wheel builds always use the 'libucxx' just built in the same CI run.
#
# env variable 'PIP_CONSTRAINT' is set up by rapids-init-pip. It constrains all subsequent
# 'pip install', 'pip download', etc. calls (except those used in 'pip wheel', handled separately in build scripts)
LIBUCXX_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libucxx ucxx --cuda "$RAPIDS_CUDA_VERSION")")
echo "libucxx-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBUCXX_WHEELHOUSE}"/libucxx_*.whl)" >> "${PIP_CONSTRAINT}"

export SKBUILD_CMAKE_ARGS="-DFIND_UCXX_CPP=ON;-DCMAKE_INSTALL_LIBDIR=ucxx/lib64;-DCMAKE_INSTALL_INCLUDEDIR=ucxx/include"

# TODO: move this variable into `ci-wheel`
# Format Python limited API version string
RAPIDS_PY_API="cp${RAPIDS_PY_VERSION//./}"
export RAPIDS_PY_API

./ci/build_wheel.sh "${package_name}" "${package_dir}" --stable

python -m auditwheel repair \
    --exclude "libucp.so.0" \
    --exclude "libucxx.so" \
    -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
    ${package_dir}/dist/*

./ci/validate_wheel.sh "${package_dir}" "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

RAPIDS_PACKAGE_NAME="$(rapids-artifact-name wheel_python ucxx ucxx --stable --cuda "$RAPIDS_CUDA_VERSION")"
export RAPIDS_PACKAGE_NAME
