#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

TIMEOUT_TOOL_PATH="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/timeout_with_stack.py

source "$(dirname "$0")/test_common.sh"

configure_ucx_tls

# Support invoking run_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

install_distributed_dev_mode() {
  # Running Distributed tests which access its internals requires installing it in
  # developer mode. This isn't a great solution but it's what we can currently do
  # to run non-public API tests in CI.

  log_message "Install Distributed in developer mode"
  MAX_ATTEMPTS=5
  for attempt in $(seq 1 $MAX_ATTEMPTS); do
    DISTRIBUTED_VERSION=$(pip list | grep "^distributed " | awk '{print $2}')

    rm -rf /tmp/distributed

    if git clone https://github.com/dask/distributed /tmp/distributed -b "${DISTRIBUTED_VERSION}"; then
      break
    else

      if [ "$attempt" -eq $MAX_ATTEMPTS ]; then
        log_message "Maximum number of attempts to clone Distributed failed."
        exit 1
      fi

      sleep 1
    fi
  done

  pip install -e /tmp/distributed
  # `pip install -e` removes files under `distributed` but not the directory if it
  # contains `__pycache__`, later causing failures to import modules.
  python -c "import os, shutil, site; \
    [shutil.rmtree(os.path.join(p, 'distributed'), ignore_errors=True) \
    for p in site.getsitepackages()]"
}

run_distributed_ucxx_tests() {
  PROGRESS_MODE=$1
  ENABLE_DELAYED_SUBMISSION=$2
  ENABLE_PYTHON_FUTURE=$3
  SKIP=$4

  CMD_LINE="UCXPY_PROGRESS_MODE=${PROGRESS_MODE} UCXPY_ENABLE_DELAYED_SUBMISSION=${ENABLE_DELAYED_SUBMISSION} UCXPY_ENABLE_PYTHON_FUTURE=${ENABLE_PYTHON_FUTURE} python ${TIMEOUT_TOOL_PATH} --enable-python $((10*60)) python -m pytest -vs python/distributed-ucxx/distributed_ucxx/tests/"

  if [ "$SKIP" -ne 0 ]; then
    echo -e "\e[1;33mSkipping unstable test: ${CMD_LINE}\e[0m"
  else
    log_command "${CMD_LINE}"
    UCXPY_PROGRESS_MODE=${PROGRESS_MODE} UCXPY_ENABLE_DELAYED_SUBMISSION=${ENABLE_DELAYED_SUBMISSION} UCXPY_ENABLE_PYTHON_FUTURE=${ENABLE_PYTHON_FUTURE} python "${TIMEOUT_TOOL_PATH}" --enable-python $((10*60)) python -m pytest -vs python/distributed-ucxx/distributed_ucxx/tests/
  fi
}

run_distributed_ucxx_tests_internal() {
  # Note that tests here require Distributed installed in developer mode!

  PROGRESS_MODE=$1
  ENABLE_DELAYED_SUBMISSION=$2
  ENABLE_PYTHON_FUTURE=$3
  SKIP=$4

  CMD_LINE="UCXPY_PROGRESS_MODE=${PROGRESS_MODE} UCXPY_ENABLE_DELAYED_SUBMISSION=${ENABLE_DELAYED_SUBMISSION} UCXPY_ENABLE_PYTHON_FUTURE=${ENABLE_PYTHON_FUTURE} python ${TIMEOUT_TOOL_PATH} --enable-python $((10*60)) python -m pytest -vs python/distributed-ucxx/distributed_ucxx/tests_internal/"

  if [ "$SKIP" -ne 0 ]; then
    echo -e "\e[1;33mSkipping unstable test: ${CMD_LINE}\e[0m"
  else
    log_command "${CMD_LINE}"
    UCXPY_PROGRESS_MODE=${PROGRESS_MODE} UCXPY_ENABLE_DELAYED_SUBMISSION=${ENABLE_DELAYED_SUBMISSION} UCXPY_ENABLE_PYTHON_FUTURE=${ENABLE_PYTHON_FUTURE} python "${TIMEOUT_TOOL_PATH}" --enable-python $((10*60)) python -m pytest -vs python/distributed-ucxx/distributed_ucxx/tests_internal/
  fi
}

# run_distributed_ucxx_tests    PROGRESS_MODE   ENABLE_DELAYED_SUBMISSION   ENABLE_PYTHON_FUTURE    SKIP
run_distributed_ucxx_tests      blocking        0                           0                       0
run_distributed_ucxx_tests      polling         0                           0                       0
run_distributed_ucxx_tests      thread          0                           0                       0
run_distributed_ucxx_tests      thread          0                           1                       0
run_distributed_ucxx_tests      thread          1                           0                       0
run_distributed_ucxx_tests      thread          1                           1                       0

install_distributed_dev_mode

# run_distributed_ucxx_tests_internal   PROGRESS_MODE   ENABLE_DELAYED_SUBMISSION   ENABLE_PYTHON_FUTURE    SKIP
run_distributed_ucxx_tests_internal     blocking        0                           0                       0
run_distributed_ucxx_tests_internal     polling         0                           0                       0
run_distributed_ucxx_tests_internal     thread          0                           0                       0
run_distributed_ucxx_tests_internal     thread          0                           1                       0
run_distributed_ucxx_tests_internal     thread          1                           0                       0
run_distributed_ucxx_tests_internal     thread          1                           1                       0
