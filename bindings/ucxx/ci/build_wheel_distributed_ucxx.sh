#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

source rapids-init-pip

package_dir="python/distributed-ucxx"

./ci/build_wheel.sh distributed-ucxx "${package_dir}"
cp "${package_dir}/dist"/* "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}/"
./ci/validate_wheel.sh "${package_dir}" "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

RAPIDS_PACKAGE_NAME="$(rapids-artifact-name wheel_python distributed-ucxx ucxx --pure --cuda "${RAPIDS_CUDA_VERSION}" --arch any)"
export RAPIDS_PACKAGE_NAME
