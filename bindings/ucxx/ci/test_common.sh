#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail


log_command() {
  CMD_LINE=$1
  echo -e "\e[1mRunning: \n ${CMD_LINE}\e[0m"
}

log_message() {
  MESSAGE=$1
  color_format="\e[1;32m"
  disable_format="\e[0m"
  echo -e "${color_format}${MESSAGE}${disable_format}"
}

print_system_stats() {
  rapids-logger "Number of CPU cores"
  nproc --all

  rapids-logger "Check GPU usage"
  # CPU-only jobs run in containers without nvidia-smi.
  command -v nvidia-smi > /dev/null && nvidia-smi || echo "(no GPU)"

  rapids-logger "Check NICs"
  awk 'END{print $1}' /etc/hosts
  cat /etc/hosts

  rapids-logger "Check maximum number of pending connections (somaxconn)"
  cat /proc/sys/net/core/somaxconn

  print_node_health
}

print_node_health() {
  # Report-only: helps attributing failures to degraded CI nodes, never fails the job.
  rapids-logger "Node health"
  python "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/node_health.py" || true
}

overlay_external_ucx() {
  # With UCXX_NO_LIBUCX=true the conda environment carries no ucx package: copy
  # the UCX installation at ucx_ROOT into the active environment. A released
  # ucx package in the same environment would shadow the copy, so fail on it.
  [[ "${UCXX_NO_LIBUCX:-false}" == "true" ]] || return 0

  cp -a "${ucx_ROOT:?}/lib/." "${CONDA_PREFIX}/lib/"
  cp -a "${ucx_ROOT}/bin/." "${CONDA_PREFIX}/bin/"
  if conda list | grep -E "^ucx +[0-9]" > /dev/null; then
    echo "ERROR: released ucx package present in the environment - it would shadow ${ucx_ROOT}" >&2
    exit 1
  fi
  echo "UCX from ${ucx_ROOT} overlaid into ${CONDA_PREFIX}"
}

print_ucx_config() {
  rapids-logger "UCX Version and Build Configuration"

  ucx_info -v
}

configure_ucx_tls() {
  local ucx_version

  if command -v ucx_info > /dev/null; then
    ucx_version="$(ucx_info -v | sed -nE 's/^# Library version: ([0-9]+\.[0-9]+\.[0-9]+).*/\1/p' | head -n 1)"
  else
    # ucx_info is not always available (wheels do not ship executables). Rely on
    # Python to determine UCX version as a fallback.
    ucx_version="$(python -c 'import ucxx; print(ucxx.__ucx_version__)')"
  fi

  # See https://github.com/rapidsai/ucxx/issues/733
  # TODO: Remove when UCX 1.23 or higher becomes minimum supported.
  if [[ "${ucx_version}" == 1.21.* ]]; then
    export UCX_TLS=^cuda_ipc
  fi
}

run_port_retry() {
  MAX_ATTEMPTS=${1}
  RUN_TYPE=${2}
  PROGRESS_MODE=${3}
  RUN_FUNCTION=${4}

  set +e
  for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
    echo "Attempt ${attempt}/${MAX_ATTEMPTS} to run ${RUN_TYPE}"

    _SERVER_PORT=$((_SERVER_PORT + 1))    # Use different ports every time to prevent `Device is busy`

    # Call the provided function with the required arguments
    ${RUN_FUNCTION} ${_SERVER_PORT} "${PROGRESS_MODE}"

    LAST_STATUS=$?
    if [ ${LAST_STATUS} -eq 0 ]; then
      break;
    fi
    sleep 1
  done
  set -e

  if [ "${LAST_STATUS}" -ne 0 ]; then
    echo "Failure running ${RUN_TYPE} after ${MAX_ATTEMPTS} attempts"
    exit "$LAST_STATUS"
  fi
}
