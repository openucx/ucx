#
# Copyright (C) Intel Corporation, 2025. All rights reserved.
# See file LICENSE for terms.
#

UCX_CHECK_GAUDI

AS_IF([test "x$gaudi_happy" = "xyes"], [uct_modules="${uct_modules}:gaudi"])

AC_ARG_ENABLE([gaudi-topo-api],
              [AS_HELP_STRING([--enable-gaudi-topo-api], 
              [Build Gaudi topology provider (experimental)])],
              [enable_gaudi_topo_api=$enableval],
              [enable_gaudi_topo_api=no])

AM_CONDITIONAL([ENABLE_GAUDI_TOPO_API], 
               [test "x$gaudi_happy" = "xyes" && test "x$enable_gaudi_topo_api" = "xyes"])

AC_CONFIG_FILES([src/uct/gaudi/Makefile
                 src/uct/gaudi/ucx-gaudi.pc])

AS_IF([test "x$gaudi_happy" = "xyes" && test "x$enable_gaudi_topo_api" = "xyes"], [
    AC_DEFINE([HAVE_GAUDI_TOPO_API], [1], [Gaudi topology API enabled])
])
