# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#
# Run a command with its output capped, so a runaway test cannot fill the
# Jenkins disk. Past the cap, the command fails.
set -o pipefail
cap=${CI_OUTPUT_CAP:-1073741824}
"$@" 2>&1 | { stdbuf -o0 head -c "${cap}"
              ! IFS= read -r -n 1 _ ||
                  { echo "ERROR: step output passed ${cap} bytes"; exit 1; }; }
