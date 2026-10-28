# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#
# Source at the start of a step: caps its output, so a runaway test cannot
# fill the Jenkins disk. Past the cap, the step fails.
ci_output_cap=${CI_OUTPUT_CAP:-1073741824}
exec {ci_stdout}>&1
exec > >(stdbuf -o0 head -c "${ci_output_cap}"
         if IFS= read -r -n 1 _; then
             echo "ERROR: step output passed ${ci_output_cap} bytes" >&${ci_stdout}
             kill -TERM $$
         fi) 2>&1
ci_output_pid=$!
# Lets the last lines reach the log before the step ends.
trap 'exec >&- 2>&-; for _ in {1..50}; do kill -0 ${ci_output_pid} 2>/dev/null || break; sleep 0.1; done' EXIT
