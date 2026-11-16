#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

TIMEOUT_TOOL_PATH="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/timeout_with_stack.py

source "$(dirname "$0")/test_common.sh"

configure_ucx_tls

# Support invoking run_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

run_py_tests() {
  if [ "${DISABLE_CYTHON:-0}" -eq 0 ]; then
    ARGS=("--run-cython")
  else
    ARGS=()
  fi

  CMD_LINE="python ${TIMEOUT_TOOL_PATH} --enable-python $((4*60)) python -m pytest --import-mode=append -vs python/ucxx/ucxx/_lib/tests/ ${ARGS[*]}"
  log_command "${CMD_LINE}"
  # Without append import mode, tests that create subprocess fail
  python "${TIMEOUT_TOOL_PATH}" --enable-python $((4*60)) \
    python -m pytest -n 4 --import-mode=append -vs python/ucxx/ucxx/_lib/tests/ "${ARGS[@]}"
}

run_py_tests_async() {
  PROGRESS_MODE=$1
  ENABLE_DELAYED_SUBMISSION=$2
  ENABLE_PYTHON_FUTURE=$3
  SKIP=$4

  CMD_LINE="UCXPY_PROGRESS_MODE=${PROGRESS_MODE} UCXPY_ENABLE_DELAYED_SUBMISSION=${ENABLE_DELAYED_SUBMISSION} UCXPY_ENABLE_PYTHON_FUTURE=${ENABLE_PYTHON_FUTURE} python ${TIMEOUT_TOOL_PATH} --enable-python $((30*60)) python -m pytest --import-mode=append -vs python/ucxx/ucxx/_lib_async/tests/ --runslow"

  if [ "$SKIP" -ne 0 ]; then
    echo -e "\e[1;33mSkipping unstable test: ${CMD_LINE}\e[0m"
  else
    log_command "${CMD_LINE}"
    # Without append import mode, tests that create subprocess fail
    UCXPY_PROGRESS_MODE=${PROGRESS_MODE} \
    UCXPY_ENABLE_DELAYED_SUBMISSION=${ENABLE_DELAYED_SUBMISSION} \
    UCXPY_ENABLE_PYTHON_FUTURE=${ENABLE_PYTHON_FUTURE} \
    python "${TIMEOUT_TOOL_PATH}" --enable-python $((30*60)) \
    python -m pytest -n 4 --import-mode=append -vs python/ucxx/ucxx/_lib_async/tests/ --runslow
  fi
}

run_py_benchmark() {
  BACKEND=$1
  PROGRESS_MODE=$2
  ASYNCIO_WAIT=$3
  ENABLE_DELAYED_SUBMISSION=$4
  ENABLE_PYTHON_FUTURE=$5
  N_BUFFERS=$6
  SLOW=$7

  if [ "$ASYNCIO_WAIT" -ne 0 ]; then
    ASYNCIO_WAIT="--asyncio-wait"
  else
    ASYNCIO_WAIT=""
  fi

  CMD_LINE="UCXPY_ENABLE_DELAYED_SUBMISSION=${ENABLE_DELAYED_SUBMISSION} UCXPY_ENABLE_PYTHON_FUTURE=${ENABLE_PYTHON_FUTURE} python ${TIMEOUT_TOOL_PATH} --enable-python $((2*60)) python -m ucxx.benchmarks.send_recv --backend ${BACKEND} -o cupy --reuse-alloc -n 8MiB --n-buffers $N_BUFFERS --progress-mode ${PROGRESS_MODE} ${ASYNCIO_WAIT}"

  # Workaround for https://github.com/rapidsai/ucxx/issues/15
  CMD_LINE="UCX_KEEPALIVE_INTERVAL=1ms ${CMD_LINE}"

  log_command "${CMD_LINE}"
  if [ "$SLOW" -ne 0 ]; then
    echo -e "\e[1;33mSLOW BENCHMARK: it may seem like a deadlock but will eventually complete.\e[0m"
  fi

  MAX_ATTEMPTS=3
  LAST_STATUS=0

  set +e
  for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
    echo "Attempt ${attempt}/${MAX_ATTEMPTS} to run Python benchmark"

    UCX_KEEPALIVE_INTERVAL=1ms \
    UCXPY_ENABLE_DELAYED_SUBMISSION=${ENABLE_DELAYED_SUBMISSION} \
    UCXPY_ENABLE_PYTHON_FUTURE=${ENABLE_PYTHON_FUTURE} \
    python "${TIMEOUT_TOOL_PATH}" --enable-python $((2*60)) \
    python -m ucxx.benchmarks.send_recv --backend "${BACKEND}" \
    -o cupy --reuse-alloc -n 8MiB --n-buffers "$N_BUFFERS" --progress-mode "${PROGRESS_MODE}" ${ASYNCIO_WAIT}

    LAST_STATUS=$?
    if [ "${LAST_STATUS}" -eq 0 ]; then
      break
    fi

    if [ "${attempt}" -lt "${MAX_ATTEMPTS}" ]; then
      echo "Python benchmark failed with status ${LAST_STATUS}; retrying"
      sleep 1
    fi
  done
  set -e

  if [ "${LAST_STATUS}" -ne 0 ]; then
    echo "Failure running Python benchmark after ${MAX_ATTEMPTS} attempts"
    exit "${LAST_STATUS}"
  fi
}

log_message "Python Core Tests"
run_py_tests

log_message "Python Async Tests"
# run_py_tests_async PROGRESS_MODE   ENABLE_DELAYED_SUBMISSION ENABLE_PYTHON_FUTURE SKIP
run_py_tests_async   thread          0                         0                    0
run_py_tests_async   thread          1                         1                    0
run_py_tests_async   blocking        0                         0                    0

log_message "Python Benchmarks"
# run_py_benchmark  BACKEND   PROGRESS_MODE   ASYNCIO_WAIT  ENABLE_DELAYED_SUBMISSION ENABLE_PYTHON_FUTURE NBUFFERS SLOW
run_py_benchmark    ucxx-core thread          0             0                         0                    1        0
run_py_benchmark    ucxx-core thread          1             0                         0                    1        0

for progress_mode in "blocking" "thread"; do
  for nbuf in 1 8; do
    # run_py_benchmark  BACKEND     PROGRESS_MODE     ASYNCIO_WAIT  ENABLE_DELAYED_SUBMISSION ENABLE_PYTHON_FUTURE NBUFFERS SLOW
    run_py_benchmark    ucxx-async  ${progress_mode}  0             0                         0                    ${nbuf}  0
    run_py_benchmark    ucxx-async  ${progress_mode}  0             0                         1                    ${nbuf}  0
    if [[ ${progress_mode} != "blocking" ]]; then
      # Delayed submission isn't support by blocking progress mode
      # run_py_benchmark  BACKEND     PROGRESS_MODE     ASYNCIO_WAIT  ENABLE_DELAYED_SUBMISSION ENABLE_PYTHON_FUTURE NBUFFERS SLOW
      run_py_benchmark    ucxx-async  ${progress_mode}  0             1                         0                    ${nbuf}  0
      run_py_benchmark    ucxx-async  ${progress_mode}  0             1                         1                    ${nbuf}  0
    fi
  done
done

log_message "C++ future -> Python future notifier example"
python "${TIMEOUT_TOOL_PATH}" --enable-python 60 python -m ucxx.examples.python_future_task_example
