#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

source rapids-init-pip

package_name="distributed_ucxx"

source "$(dirname "$0")/test_common.sh"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

LIBUCXX_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libucxx ucxx --cuda "$RAPIDS_CUDA_VERSION")")
UCXX_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python ucxx ucxx --stable --cuda "$RAPIDS_CUDA_VERSION")")
DISTRIBUTED_UCXX_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python distributed-ucxx ucxx --pure --cuda "$RAPIDS_CUDA_VERSION" --arch any)")

# generate constraints (possibly pinning to oldest support versions of dependencies)
rapids-generate-pip-constraints test_python "${PIP_CONSTRAINT}"

# notes:
#
#   * echo to expand wildcard before adding `[test]` requires for pip
#   * just providing --constraint="${PIP_CONSTRAINT}" to be explicit, and because
#     that environment variable is ignored if any other --constraint are passed via the CLI
#
rapids-pip-retry install \
    -v \
    --prefer-binary \
    --constraint "${PIP_CONSTRAINT}" \
    "${LIBUCXX_WHEELHOUSE}/libucxx_${RAPIDS_PY_CUDA_SUFFIX}"*.whl \
    "${UCXX_WHEELHOUSE}/ucxx_${RAPIDS_PY_CUDA_SUFFIX}"*.whl \
    "$(echo "${DISTRIBUTED_UCXX_WHEELHOUSE}"/"${package_name}_${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test]"

print_system_stats

rapids-logger "Run distributed-ucxx tests with wheels"
./ci/run_python_distributed.sh
