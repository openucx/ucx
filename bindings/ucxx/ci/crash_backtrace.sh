#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Debugger command used by UCX's error handler (`UCX_HANDLE_ERRORS=debug` with
# `UCX_GDB_COMMAND` pointing to this script, see `configure_crash_backtraces` in
# `ci/test_common.sh`). When a process that loaded UCX receives a fatal signal
# (SIGSEGV, SIGBUS, SIGILL, SIGFPE), UCX forks and executes this script with
# `-p <pid> -x <commands-file>` and waits for it to complete before resuming
# termination of the process (including Python's faulthandler output).
#
# Print native stacks of all threads of the crashing process with GDB, bounded by a
# timeout. Output goes to stderr since stdout of pytest-xdist workers is not
# forwarded to the CI log.

exec 1>&2

echo "==== UCXX CI: native backtraces of all threads (crashing thread is in ucs_error_signal_handler) ===="
timeout --kill-after=10 "${UCXX_CRASH_GDB_TIMEOUT:-300}" gdb \
  -quiet -batch -nx \
  -iex "set auto-load python-scripts off" \
  -ex "set pagination off" \
  -ex "set confirm off" \
  -ex "set print thread-events off" \
  -ex "info threads" \
  -ex "thread apply all bt" \
  "$@"
echo "==== UCXX CI: end of native backtraces (gdb exit code: $?) ===="
