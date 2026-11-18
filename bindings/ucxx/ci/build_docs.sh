#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libucxx ucxx --cuda "$RAPIDS_CUDA_VERSION")")
PYTHON_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_python ucxx ucxx --stable --cuda "$RAPIDS_CUDA_VERSION")")

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

UCXX_VERSION="$(head -1 ./VERSION)"
export UCXX_VERSION
UCXX_VERSION_MAJOR_MINOR="$(sed -E -e 's/^([0-9]+)\.([0-9]+)\.([0-9]+).*$/\1.\2/' VERSION)"

ENV_YAML_DIR="$(mktemp -d)"

rapids-dependency-file-generator \
  --output conda \
  --file-key docs \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};no_libucx=${UCXX_NO_LIBUCX:-false}" \
  --prepend-channel "${CPP_CHANNEL}" \
  --prepend-channel "${PYTHON_CHANNEL}" \
  | tee "${ENV_YAML_DIR}/env.yaml"

rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n docs
conda activate docs

# With UCXX_NO_LIBUCX=true the docs environment carries no ucx package: copy
# the UCX installation at ucx_ROOT into it.
if [[ "${UCXX_NO_LIBUCX:-false}" == "true" ]]; then
  cp -a "${ucx_ROOT:?}/lib/." "${CONDA_PREFIX}/lib/"
  cp -a "${ucx_ROOT}/bin/." "${CONDA_PREFIX}/bin/"
  echo "UCX from ${ucx_ROOT} overlaid into ${CONDA_PREFIX}"
fi

rapids-print-env

# The caller may set the output directory to collect the docs from it.
RAPIDS_DOCS_DIR="${RAPIDS_DOCS_DIR:-$(mktemp -d)}"
export RAPIDS_DOCS_DIR

rapids-logger "Build CPP docs"
pushd cpp/doxygen
doxygen Doxyfile
mkdir -p "${RAPIDS_DOCS_DIR}/libucxx/html"
mv html/* "${RAPIDS_DOCS_DIR}/libucxx/html"
popd

rapids-logger "Build Python docs"
pushd docs/ucxx
make dirhtml O="-j 8"
mkdir -p "${RAPIDS_DOCS_DIR}/ucxx/html"
mv build/dirhtml/* "${RAPIDS_DOCS_DIR}/ucxx/html"
popd

RAPIDS_VERSION_NUMBER="${UCXX_VERSION_MAJOR_MINOR}" rapids-upload-docs
