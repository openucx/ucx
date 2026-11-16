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
  nvidia-smi

  rapids-logger "Check NICs"
  awk 'END{print $1}' /etc/hosts
  cat /etc/hosts

  rapids-logger "Check maximum number of pending connections (somaxconn)"
  cat /proc/sys/net/core/somaxconn
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
