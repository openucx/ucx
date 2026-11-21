#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

TIMEOUT_TOOL_PATH="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/timeout_with_stack.py

source "$(dirname "$0")/test_common.sh"

configure_ucx_tls

# Support invoking run_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

##################################### C++ ######################################
_SERVER_PORT=12345

# First, try the installed location (CI/conda environments)
installed_test_location="${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/gtests/libucxx/"
# Fall back to the build directory (devcontainer environments)
devcontainers_test_location="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../cpp/build/latest/gtests"

if [[ -d "${installed_test_location}" ]]; then
    GTESTS_PATH="${installed_test_location}"
elif [[ -d "${devcontainers_test_location}" ]]; then
    GTESTS_PATH="${devcontainers_test_location}"
else
    echo "Error: Test location not found. Searched:" >&2
    echo "  - ${installed_test_location}" >&2
    echo "  - ${devcontainers_test_location}" >&2
    exit 1
fi

# First, try the installed location (CI/conda environments)
installed_examples_location="${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/examples/libucxx/"
# Fall back to the build directory (devcontainer environments)
devcontainers_examples_location="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../cpp/build/latest/examples"

if [[ -d "${installed_examples_location}" ]]; then
    EXAMPLES_PATH="${installed_examples_location}"
elif [[ -d "${devcontainers_examples_location}" ]]; then
    EXAMPLES_PATH="${devcontainers_examples_location}"
else
    EXAMPLES_PATH=""
fi

run_cpp_tests() {
  CMD_LINE="python ${TIMEOUT_TOOL_PATH} $((20*60)) ${GTESTS_PATH}/UCXX_TEST"

  log_command "${CMD_LINE}"
  UCX_TCP_CM_REUSEADDR=y ${CMD_LINE}
}

run_cpp_benchmarks() {
  # Run the dedicated benchmark script
  "$(dirname "$0")/run_cpp_benchmarks.sh"
}

run_cpp_example() {
  SERVER_PORT=$1
  PROGRESS_MODE=$2

  if [[ -z "${EXAMPLES_PATH}" ]]; then
    echo "Skipping examples: examples not built (BUILD_EXAMPLES=OFF)" >&2
    return 0
  fi

  CMD_LINE="python ${TIMEOUT_TOOL_PATH} --enable-python 60 ${EXAMPLES_PATH}/ucxx_example_basic -P ${PROGRESS_MODE} -p ${SERVER_PORT}"

  log_command "${CMD_LINE}"
  UCX_TCP_CM_REUSEADDR=y ${CMD_LINE}
}

log_message "C++ Tests"
run_cpp_tests

log_message "C++ Examples"
# run_port_retry MAX_ATTEMPTS RUN_TYPE PROGRESS_MODE FUNCTION_NAME
run_port_retry 10 "example" "polling" "run_cpp_example"
run_port_retry 10 "example" "blocking" "run_cpp_example"
run_port_retry 10 "example" "thread-polling" "run_cpp_example"
run_port_retry 10 "example" "thread-blocking" "run_cpp_example"
run_port_retry 10 "example" "wait" "run_cpp_example"
