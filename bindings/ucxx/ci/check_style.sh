#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

rapids-logger "Create checks conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-dependency-file-generator \
  --output conda \
  --file-key checks \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n checks
conda activate checks

RAPIDS_BRANCH="$(cat "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../RAPIDS_BRANCH)"

export RAPIDS_CMAKE_FORMAT_FILE=/tmp/rapids_cmake_ci/cmake-format-rapids-cmake.json
mkdir -p "$(dirname "${RAPIDS_CMAKE_FORMAT_FILE}")"
rapids-github-fetch rapidsai/rapids-cmake cmake-format-rapids-cmake.json "${RAPIDS_BRANCH}" > "${RAPIDS_CMAKE_FORMAT_FILE}"

# Run pre-commit checks
pre-commit run --all-files --show-diff-on-failure
