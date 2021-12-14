#
# Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

UCX_CHECK_GDRCOPY

AS_IF([test "x$gdrcopy_happy" = "xyes" && test "x$rocm_happy" = "xyes"],
      [uct_rocm_modules="${uct_rocm_modules}:gdr"])
AC_CONFIG_FILES([src/uct/rocm/gdr/Makefile
                 src/uct/rocm/gdr/ucx-rocm-gdr.pc])
