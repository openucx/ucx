#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

UCX_CHECK_CUDA
AS_IF([test "x$cuda_happy" = "xyes"], [ucm_modules="${ucm_modules}:cuda"])
AC_CONFIG_FILES([src/ucm/cuda/Makefile])
