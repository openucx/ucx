#
# Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

AC_DEFUN([PKG_CHECK_MODULES_STATIC],
         [AC_REQUIRE([PKG_PROG_PKG_CONFIG])
          save_PKG_CONFIG=$PKG_CONFIG
          PKG_CONFIG="$PKG_CONFIG --static"
          PKG_CHECK_MODULES([$1], [$2], [$3], [$4])
          PKG_CONFIG=$save_PKG_CONFIG
         ])
