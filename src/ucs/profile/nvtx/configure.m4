#
# Copyright (C) NVIDIA Corporation. 2021.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

UCX_CHECK_NVTX

AS_IF([test "x$nvtx_happy" = "xyes"], [ucs_modules="${ucs_modules}:nvtx"])
AC_CONFIG_FILES([src/ucs/profile/nvtx/Makefile])
