#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
# Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

xpmem_happy="no"

UCX_CHECK_MODULE([xpmem], [XPMEM], [XPMEM], [xpmem],
                 [[Enable the use of XPMEM], [Search XPMEM library in DIR], [-],
                  [XPMEM could not be found]],
                  [xpmem_happy=yes], [],
                  [[xpmem.h], [xpmem_version], [[#include <xpmem.h>]], [xpmem], [xpmem_init], []],
                  [/opt/xpmem, /usr/local, /usr/local/xpmem])

AS_IF([test "x$xpmem_happy" = "xyes"], [uct_modules="${uct_modules}:xpmem"])
AM_CONDITIONAL([HAVE_XPMEM], [test "x$xpmem_happy" != "xno"])
AC_SUBST(XPMEM_CFLAGS)
AC_SUBST(XPMEM_LIBS)
AC_CONFIG_FILES([src/uct/sm/mm/xpmem/Makefile
                 src/uct/sm/mm/xpmem/ucx-xpmem.pc])
