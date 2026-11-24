# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
# cmake-format: on
# =============================================================================

# This function finds cccl and sets any additional necessary environment variables.
function(find_and_configure_cccl)
  include(${rapids-cmake-dir}/cpm/cccl.cmake)

  # Find or install CCCL
  rapids_cpm_cccl(BUILD_EXPORT_SET ucxx-exports INSTALL_EXPORT_SET ucxx-exports)

endfunction()

find_and_configure_cccl()
