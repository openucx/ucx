#
# Copyright (C) Intel Corporation, 2025. All rights reserved.
#
# See file LICENSE for terms.
#

UCX_CHECK_GAUDI

AS_IF([test "x$gaudi_happy" = "xyes"], [uct_modules="${uct_modules}:gaudi"])
uct_gaudi_modules=""
AC_DEFINE_UNQUOTED([uct_gaudi_MODULES], ["${uct_gaudi_modules}"], [GAUDI loadable modules])
AC_CONFIG_FILES([src/uct/gaudi/Makefile
                 src/uct/gaudi/ucx-gaudi.pc])
