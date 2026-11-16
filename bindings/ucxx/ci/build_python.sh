#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

source rapids-configure-sccache
source rapids-datetime-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin py build"

CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libucxx ucxx --cuda "$RAPIDS_CUDA_VERSION")")

sccache --stop-server 2>/dev/null || true

UCXX_PACKAGE_VERSION=$(rapids-generate-version)
export UCXX_PACKAGE_VERSION
RAPIDS_PACKAGE_DEPENDENCY=$(sed -E -e 's/^([0-9]+\.)0?([1-9][0-9]?)\.[0-9]+$/\1\2.*/' RAPIDS_VERSION)
export RAPIDS_PACKAGE_DEPENDENCY

# Creates and exports $RATTLER_CHANNELS
source rapids-rattler-channel-string

# Creates artifacts directory for telemetry
source rapids-telemetry-setup

# --no-build-id allows for caching with `sccache`
# more info is available at
# https://rattler.build/latest/tips_and_tricks/#using-sccache-or-ccache-with-rattler-build
rapids-telemetry-record build.log rattler-build build \
    --recipe conda/recipes/ucxx \
    --experimental \
    --no-build-id \
    --output-dir "$RAPIDS_CONDA_BLD_OUTPUT_DIR" \
    -c "${CPP_CHANNEL}" \
    "${RATTLER_CHANNELS[@]}"

rapids-telemetry-record sccache-stats.txt sccache --show-adv-stats
sccache --stop-server >/dev/null 2>&1 || true

# See https://github.com/prefix-dev/rattler-build/issues/1424
rm -rf "$RAPIDS_CONDA_BLD_OUTPUT_DIR"/build_cache

RAPIDS_PACKAGE_NAME="$(rapids-artifact-name conda_python ucxx ucxx --stable --cuda)"
export RAPIDS_PACKAGE_NAME
