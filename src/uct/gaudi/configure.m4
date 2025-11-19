#
# Copyright (C) Intel Corporation, 2025. All rights reserved.
#
# See file LICENSE for terms.
#

UCX_CHECK_GAUDI

AS_IF([test "x$gaudi_happy" = "xyes"], [uct_modules="${uct_modules}:gaudi"])
AC_CONFIG_FILES([src/uct/gaudi/Makefile
                 src/uct/gaudi/ucx-gaudi.pc])
