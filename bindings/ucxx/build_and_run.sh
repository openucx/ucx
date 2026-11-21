#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

NUMARGS=$#
ARGS=$*

# NOTE: ensure all dir changes are relative to the location of this
# script, and that this script resides in the repo dir!
REPODIR=$(cd "$(dirname "$0")"; pwd)

VALIDARGS="cpp_tests py_tests cpp_examples py_async_tests py_bench py_async_bench cython_tests -v -g -n -c --show_depr_warn -h"
HELP="$0 [cpp_tests] [cpp_bench] [cpp_examples] [py_tests] [py_async_tests] [py_bench] [py_async_bench] [cython_tests]
   cpp_tests                     - run all C++ tests
   cpp_bench                     - run C++ benchmarks
   cpp_examples                  - run C++ examples
   py_tests                      - run all Python core tests
   py_async_tests                - run all Python async tests
   py_bench                      - run Python core benchmarks
   py_async_bench                - run Python async benchmarks
   cython_tests                  - run all Python tests of public Cython API
   clean                         - remove all existing build artifacts and configuration (start
                                   over)
   -v                            - verbose build mode
   -g                            - build for debug
   -n                            - no install step
   -c                            - create cpp/compile_commands.json
   --show_depr_warn              - show cmake deprecation warnings
   --cmake-args=\\\"<args>\\\"   - pass arbitrary list of CMake configuration options (escape all quotes in argument)
   -h | --h[elp]                 - print this text

   default action (no args) is to build (with command below) and run all tests and benchmarks.
     ./build.sh
"
BUILD_ARGS=""

RUN_CPP_TESTS=0
RUN_CPP_BENCH=0
RUN_CPP_EXAMPLE=0
RUN_PY_TESTS=0
RUN_PY_ASYNC_TESTS=0
RUN_PY_BENCH=0
RUN_PY_ASYNC_BENCH=0
RUN_CYTHON_TESTS=0

BINARY_PATH=${CONDA_PREFIX}/bin

function hasArg {
    (( NUMARGS != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

function runAll {
    (( NUMARGS == 0 )) || ! (echo " ${ARGS} " | grep -q " [^-]\+ ")
}

if hasArg -h || hasArg --h || hasArg --help; then
    echo "${HELP}"
    exit 0
fi

# Check for valid usage
if (( NUMARGS != 0 )); then
    for a in ${ARGS}; do
    if ! (echo " ${VALIDARGS} " | grep -q " ${a} "); then
        echo "Invalid option or formatting, check --help: ${a}"
        exit 1
    fi
    done
fi

if hasArg -v; then
    BUILD_ARGS="${BUILD_ARGS} -v"
fi
if hasArg -g; then
    BUILD_ARGS="${BUILD_ARGS} -g"
fi
if hasArg -n; then
    BUILD_ARGS="${BUILD_ARGS} -n"
fi
if hasArg -c; then
    BUILD_ARGS="${BUILD_ARGS} -c"
fi

if runAll || hasArg cpp_tests; then
    RUN_CPP_TESTS=1
    BUILD_ARGS="${BUILD_ARGS} libucxx libucxx_tests"
fi
if runAll || hasArg cpp_bench; then
    RUN_CPP_BENCH=1
    BUILD_ARGS="${BUILD_ARGS} libucxx libucxx_benchmarks"
fi
if runAll || hasArg cpp_examples; then
    RUN_CPP_EXAMPLE=1
    BUILD_ARGS="${BUILD_ARGS} libucxx libucxx_examples"
fi
if runAll || hasArg py_tests; then
    RUN_PY_TESTS=1
fi
if runAll || hasArg py_async_tests; then
    RUN_PY_ASYNC_TESTS=1
fi
if runAll || hasArg py_bench; then
    RUN_PY_BENCH=1
fi
if runAll || hasArg py_async_bench; then
    RUN_PY_ASYNC_BENCH=1
fi
if runAll || hasArg cython_tests; then
    RUN_CYTHON_TESTS=1
fi

# Exit if a building error occurs
set -e

(
  cd "${REPODIR}"
  ./build.sh "${BUILD_ARGS}"
)

# Let all tests run even if they fail
set +e

run_cpp_benchmark() {
  PROGRESS_MODE=$1

  # UCX_TCP_CM_REUSEADDR=y to be able to bind immediately to the same port before
  # `TIME_WAIT` timeout
  CMD_LINE_SERVER="UCX_TCP_CM_REUSEADDR=y ${BINARY_PATH}/benchmarks/libucxx/ucxx_perftest -s 8388608 -r -n 20 -P ${PROGRESS_MODE}"
  CMD_LINE_CLIENT="${BINARY_PATH}/benchmarks/libucxx/ucxx_perftest -s 8388608 -r -n 20 -P ${PROGRESS_MODE} 127.0.0.1"

  echo -e "\e[1mRunning: \n  - ${CMD_LINE_SERVER}\n  - ${CMD_LINE_CLIENT}\e[0m"
  UCX_TCP_CM_REUSEADDR=y "${BINARY_PATH}/benchmarks/libucxx/ucxx_perftest" -s 8388608 -r -n 20 -P "${PROGRESS_MODE}" &
  "${BINARY_PATH}/benchmarks/libucxx/ucxx_perftest" -s 8388608 -r -n 20 -P "${PROGRESS_MODE}" 127.0.0.1
}

run_cpp_example() {
  PROGRESS_MODE=$1
  SEND_BUFFER_TYPE=$2
  RECV_BUFFER_TYPE=$3

  # UCX_TCP_CM_REUSEADDR=y to be able to bind immediately to the same port before
  # `TIME_WAIT` timeout
  CMD_LINE="UCX_TCP_CM_REUSEADDR=y ${BINARY_PATH}/examples/libucxx/ucxx_example_basic -P ${PROGRESS_MODE} -s ${SEND_BUFFER_TYPE} -r ${RECV_BUFFER_TYPE}"

  echo -e "\e[1mRunning: \n  - ${CMD_LINE}\e[0m"
  UCX_TCP_CM_REUSEADDR=y "${BINARY_PATH}/examples/libucxx/ucxx_example_basic" -P "${PROGRESS_MODE}" -s "${SEND_BUFFER_TYPE}" -r "${RECV_BUFFER_TYPE}"
}

run_tests_async() {
  PROGRESS_MODE=$1
  ENABLE_DELAYED_SUBMISSION=$2
  ENABLE_PYTHON_FUTURE=$3
  SKIP=$4

  CMD_LINE="UCXPY_PROGRESS_MODE=${PROGRESS_MODE} UCXPY_ENABLE_DELAYED_SUBMISSION=${ENABLE_DELAYED_SUBMISSION} UCXPY_ENABLE_PYTHON_FUTURE=${ENABLE_PYTHON_FUTURE} pytest -vs python/ucxx/ucxx/_lib_async/tests/"

  if [ "$SKIP" -ne 0 ]; then
    echo -e "\e[31;1mSkipping unstable test: ${CMD_LINE}\e[0m"
  else
    echo -e "\e[1mRunning: ${CMD_LINE}\e[0m"
    UCXPY_PROGRESS_MODE=${PROGRESS_MODE} UCXPY_ENABLE_DELAYED_SUBMISSION=${ENABLE_DELAYED_SUBMISSION} UCXPY_ENABLE_PYTHON_FUTURE=${ENABLE_PYTHON_FUTURE} pytest -vs python/ucxx/ucxx/_lib_async/tests/
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

  CMD_LINE="UCXPY_ENABLE_DELAYED_SUBMISSION=${ENABLE_DELAYED_SUBMISSION} UCXPY_ENABLE_PYTHON_FUTURE=${ENABLE_PYTHON_FUTURE} python -m ucxx.benchmarks.send_recv --backend ${BACKEND} -o cupy --reuse-alloc -d 0 -e 1 -n 8MiB --n-buffers $N_BUFFERS --progress-mode ${PROGRESS_MODE} ${ASYNCIO_WAIT}"

  echo -e "\e[1mRunning: ${CMD_LINE}\e[0m"
  if [ "$SLOW" -ne 0 ]; then
    echo -e "\e[31;1mSLOW BENCHMARK: it may seem like a deadlock but will eventually complete.\e[0m"
  fi
  UCXPY_ENABLE_DELAYED_SUBMISSION=${ENABLE_DELAYED_SUBMISSION} \
    UCXPY_ENABLE_PYTHON_FUTURE=${ENABLE_PYTHON_FUTURE} \
    python -m ucxx.benchmarks.send_recv --backend "${BACKEND}" -o cupy \
    --reuse-alloc -d 0 -e 1 -n 8MiB --n-buffers "$N_BUFFERS" --progress-mode "${PROGRESS_MODE}" ${ASYNCIO_WAIT}
}

if [[ $RUN_CPP_TESTS != 0 ]]; then
  # UCX_TCP_CM_REUSEADDR=y to be able to bind immediately to the same port before
  # `TIME_WAIT` timeout
  UCX_TCP_CM_REUSEADDR=y "${BINARY_PATH}/gtests/libucxx/UCXX_TEST"
fi
if [[ $RUN_CPP_BENCH != 0 ]]; then
  # run_cpp_benchmark PROGRESS_MODE
  run_cpp_benchmark   polling
  run_cpp_benchmark   blocking
  run_cpp_benchmark   thread-polling
  run_cpp_benchmark   thread-blocking
  run_cpp_benchmark   wait
fi
if [[ $RUN_CPP_EXAMPLE != 0 ]]; then
  for send_buffer_type in host rmm; do
    for recv_buffer_type in host rmm; do
      # run_cpp_example PROGRESS_MODE   SEND_BUFFER_TYPE    RECV_BUFFER_TYPE
      run_cpp_example   polling         ${send_buffer_type} ${recv_buffer_type}
      run_cpp_example   blocking        ${send_buffer_type} ${recv_buffer_type}
      run_cpp_example   thread-polling  ${send_buffer_type} ${recv_buffer_type}
      run_cpp_example   thread-blocking ${send_buffer_type} ${recv_buffer_type}
      run_cpp_example   wait            ${send_buffer_type} ${recv_buffer_type}
    done
  done
fi
if [[ $RUN_PY_TESTS != 0 ]]; then
  if [ $RUN_CYTHON_TESTS -ne 0 ]; then
    ARGS="--run-cython"
  else
    ARGS=""
  fi

  echo -e "\e[1mRunning: pytest-vs python/ucxx/ucxx/_lib/tests/ ${ARGS}\e[0m"
  pytest -vs python/ucxx/ucxx/_lib/tests/ ${ARGS}
fi
if [[ $RUN_PY_ASYNC_TESTS != 0 ]]; then
  # run_tests_async PROGRESS_MODE   ENABLE_DELAYED_SUBMISSION ENABLE_PYTHON_FUTURE SKIP
  run_tests_async   polling         0                         0                    0
  run_tests_async   polling         0                         1                    0
  run_tests_async   polling         1                         0                    1    # Delayed submission can't be used with polling
  run_tests_async   polling         1                         1                    1    # Delayed submission can't be used with polling
  run_tests_async   thread-polling  0                         0                    0
  run_tests_async   thread-polling  0                         1                    0
  run_tests_async   thread-polling  1                         0                    0
  run_tests_async   thread-polling  1                         1                    0
  run_tests_async   thread          0                         0                    0
  run_tests_async   thread          0                         1                    0
  run_tests_async   thread          1                         0                    0
  run_tests_async   thread          1                         1                    0
fi

if [[ $RUN_PY_BENCH != 0 ]]; then
  # run_py_benchmark  BACKEND   PROGRESS_MODE   ASYNCIO_WAIT  ENABLE_DELAYED_SUBMISSION ENABLE_PYTHON_FUTURE NBUFFERS SLOW
  run_py_benchmark    ucxx-core blocking        0             0                         0                    1        0
  run_py_benchmark    ucxx-core polling         0             0                         0                    1        0
  run_py_benchmark    ucxx-core thread-polling  0             0                         0                    1        0
  run_py_benchmark    ucxx-core thread-polling  1             0                         0                    1        0
  run_py_benchmark    ucxx-core thread          0             0                         0                    1        0
  run_py_benchmark    ucxx-core thread          1             0                         0                    1        0
fi
if [[ $RUN_PY_ASYNC_BENCH != 0 ]]; then
  for nbuf in 1 8; do
    # run_py_benchmark  BACKEND     PROGRESS_MODE   ASYNCIO_WAIT  ENABLE_DELAYED_SUBMISSION ENABLE_PYTHON_FUTURE NBUFFERS SLOW
    run_py_benchmark    ucxx-async  polling         0             0                         0                    ${nbuf}  0
    run_py_benchmark    ucxx-async  polling         0             0                         1                    ${nbuf}  0
    run_py_benchmark    ucxx-async  polling         0             1                         0                    ${nbuf}  0
    run_py_benchmark    ucxx-async  polling         0             1                         1                    ${nbuf}  0
    run_py_benchmark    ucxx-async  thread-polling  0             0                         0                    ${nbuf}  0
    run_py_benchmark    ucxx-async  thread-polling  0             0                         1                    ${nbuf}  0
    run_py_benchmark    ucxx-async  thread-polling  0             1                         0                    ${nbuf}  0
    run_py_benchmark    ucxx-async  thread-polling  0             1                         1                    ${nbuf}  0
    if [ ${nbuf} -eq 1 ]; then
      run_py_benchmark    ucxx-async  thread          0             0                         0                    ${nbuf}  1
      run_py_benchmark    ucxx-async  thread          0             0                         1                    ${nbuf}  1
      run_py_benchmark    ucxx-async  thread          0             1                         0                    ${nbuf}  1
    else
      run_py_benchmark    ucxx-async  thread          0             0                         0                    ${nbuf}  0
      run_py_benchmark    ucxx-async  thread          0             0                         1                    ${nbuf}  0
      run_py_benchmark    ucxx-async  thread          0             1                         0                    ${nbuf}  0
    fi
    run_py_benchmark    ucxx-async  thread          0             1                         1                    ${nbuf}  0
  done
fi
