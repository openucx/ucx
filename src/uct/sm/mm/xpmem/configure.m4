#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
# Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

xpmem_happy="no"

UCX_CHECK_MODULE([xpmem], [XPMEM], [XPMEM], [xpmem],
                 [[], [], [-], []],
                  [have_xpmem=true
                   uct_modules="${uct_modules}:xpmem"], [have_xpmem=false],
                  [[xpmem.h], [xpmem_version], [[#include <xpmem.h>]], [xpmem], [xpmem_init], []],
                  [/opt/xpmem, /usr/local, /usr/local/xpmem])

AM_CONDITIONAL([HAVE_XPMEM], [$have_xpmem])
AC_SUBST(XPMEM_CFLAGS)
AC_SUBST(XPMEM_LIBS)
AC_CONFIG_FILES([src/uct/sm/mm/xpmem/Makefile
                 src/uct/sm/mm/xpmem/ucx-xpmem.pc])
