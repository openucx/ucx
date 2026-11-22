#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

TIMEOUT_TOOL_PATH="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/timeout_with_stack.py

source "$(dirname "$0")/test_common.sh"

configure_ucx_tls

# Support invoking run_cpp_benchmarks.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

##################################### C++ Benchmarks ######################################
_SERVER_PORT=12345

run_cpp_benchmark() {
  SERVER_PORT=$1
  PROGRESS_MODE=$2
  TEST_TYPE=${3:-"tag_lat"}
  MEMORY_TYPE=${4:-"host"}
  MESSAGE_SIZE=${5:-"8388608"}
  NUM_ITERATIONS=${6:-"20"}
  NUM_WARMUP=${7:-"3"}
  PERCENTILE_RANK=${8:-"50.0"}
  ENDPOINT_ERROR_HANDLING=${9:-""}
  REUSE_ALLOCATIONS=${10:-""}
  VERIFY_RESULTS=${11:-""}

  RUNTIME_PATH=${CONDA_PREFIX:-./}
  BINARY_PATH=${RUNTIME_PATH}/bin

  # Build command line with all options
  CMD_BASE="${BINARY_PATH}/benchmarks/libucxx/ucxx_perftest"
  CMD_OPTS="-t ${TEST_TYPE} -m ${MEMORY_TYPE} -P ${PROGRESS_MODE} -p ${SERVER_PORT} -s ${MESSAGE_SIZE} -n ${NUM_ITERATIONS} -w ${NUM_WARMUP} -R ${PERCENTILE_RANK}"

  # Add optional flags
  if [[ "${ENDPOINT_ERROR_HANDLING}" == "true" ]]; then
    CMD_OPTS="${CMD_OPTS} -e"
  fi
  if [[ "${REUSE_ALLOCATIONS}" == "false" ]]; then
    CMD_OPTS="${CMD_OPTS} -L"
  fi
  if [[ "${VERIFY_RESULTS}" == "true" ]]; then
    CMD_OPTS="${CMD_OPTS} -v"
  fi

  CMD_LINE_SERVER="python ${TIMEOUT_TOOL_PATH} 60 ${CMD_BASE} ${CMD_OPTS}"
  CMD_LINE_CLIENT="python ${TIMEOUT_TOOL_PATH} 60 ${CMD_BASE} ${CMD_OPTS} 127.0.0.1"

  log_command "${CMD_LINE_SERVER}"
  UCX_TCP_CM_REUSEADDR=y ${CMD_LINE_SERVER} &
  sleep 1

  log_command "${CMD_LINE_CLIENT}"
  ${CMD_LINE_CLIENT}
}

run_cpp_benchmark_comprehensive() {
  SERVER_PORT=$1
  PROGRESS_MODE=$2

  # Test different combinations systematically

  # Basic latency test with default settings
  run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "host" "8388608" "20" "3" "50.0" "" "" ""

  # Test with different message sizes for latency
  run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "host" "8" "20" "3" "50.0" "" "" ""
  run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "host" "1024" "20" "3" "50.0" "" "" ""
  run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "host" "1048576" "20" "3" "50.0" "" "" ""

  # Test bandwidth test
  run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_bw" "host" "8388608" "20" "3" "50.0" "" "" ""

  # Test with different percentile ranks
  run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "host" "8388608" "20" "3" "95.0" "" "" ""
  run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "host" "8388608" "20" "3" "99.0" "" "" ""

  # Test with endpoint error handling
  run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "host" "8388608" "20" "3" "50.0" "true" "" ""

  # Test with disabled reuse allocations
  run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "host" "8388608" "20" "3" "50.0" "" "false" ""

  # Test with verification enabled
  run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "host" "8388608" "20" "3" "50.0" "" "" "true"

  # Test with different iteration counts
  run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "host" "8388608" "50" "5" "50.0" "" "" ""

  # Test with different warmup counts
  run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "host" "8388608" "20" "10" "50.0" "" "" ""
}

run_cpp_benchmark_cuda_tests() {
  SERVER_PORT=$1
  PROGRESS_MODE=$2

  # Only run CUDA tests if CUDA is available
  if command -v nvidia-smi &> /dev/null; then
    log_message "Running CUDA memory tests for progress mode: ${PROGRESS_MODE}"

    # Test CUDA memory types
    run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "cuda" "8388608" "20" "3" "50.0" "" "" ""
    run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_bw" "cuda" "8388608" "20" "3" "50.0" "" "" ""

    # Test CUDA managed memory
    run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "cuda-managed" "8388608" "20" "3" "50.0" "" "" ""
    run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_bw" "cuda-managed" "8388608" "20" "3" "50.0" "" "" ""

    # Test CUDA async memory
    run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "cuda-async" "8388608" "20" "3" "50.0" "" "" ""
    run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_bw" "cuda-async" "8388608" "20" "3" "50.0" "" "" ""

    # Test with smaller message sizes for CUDA
    run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "cuda" "1024" "20" "3" "50.0" "" "" ""
    run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "cuda-managed" "1024" "20" "3" "50.0" "" "" ""
  else
    log_message "Skipping CUDA tests - CUDA not available"
  fi
}

run_cpp_benchmark_edge_cases() {
  SERVER_PORT=$1
  PROGRESS_MODE=$2

  # Test edge cases and combinations

  # Very small message size
  run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "host" "1" "20" "3" "50.0" "" "" ""

  # Large message size
  run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "host" "33554432" "10" "3" "50.0" "" "" ""

  # High percentile rank
  run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "host" "8388608" "20" "3" "99.9" "" "" ""

  # Many iterations
  run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "host" "8388608" "100" "10" "50.0" "" "" ""

  # All flags enabled
  run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_lat" "host" "8388608" "20" "3" "50.0" "true" "false" "true"

  # Bandwidth test with different sizes
  run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_bw" "host" "1048576" "20" "3" "50.0" "" "" ""
  run_cpp_benchmark "${SERVER_PORT}" "${PROGRESS_MODE}" "tag_bw" "host" "16777216" "20" "3" "50.0" "" "" ""
}



log_message "C++ Benchmarks - Basic Tests"
# Basic tests for each progress mode
run_port_retry 10 "benchmark" "polling" "run_cpp_benchmark"
run_port_retry 10 "benchmark" "blocking" "run_cpp_benchmark"
run_port_retry 10 "benchmark" "thread-polling" "run_cpp_benchmark"
run_port_retry 10 "benchmark" "thread-blocking" "run_cpp_benchmark"
run_port_retry 10 "benchmark" "wait" "run_cpp_benchmark"

log_message "C++ Benchmarks - Comprehensive Tests"
# Comprehensive tests for a subset of progress modes (to avoid too many tests)
run_port_retry 10 "benchmark_comprehensive" "polling" "run_cpp_benchmark_comprehensive"
run_port_retry 10 "benchmark_comprehensive" "blocking" "run_cpp_benchmark_comprehensive"

log_message "C++ Benchmarks - CUDA Tests"
# CUDA tests for a subset of progress modes
run_port_retry 10 "benchmark_cuda" "polling" "run_cpp_benchmark_cuda_tests"
run_port_retry 10 "benchmark_cuda" "blocking" "run_cpp_benchmark_cuda_tests"

log_message "C++ Benchmarks - Edge Cases"
# Edge case tests for polling mode only (to keep test time reasonable)
run_port_retry 10 "benchmark_edge_cases" "polling" "run_cpp_benchmark_edge_cases"
