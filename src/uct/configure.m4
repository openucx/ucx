#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

uct_modules=""
m4_include([src/uct/cuda/configure.m4])
m4_include([src/uct/ib/configure.m4])
m4_include([src/uct/rocm/configure.m4])
m4_include([src/uct/sm/configure.m4])
m4_include([src/uct/ugni/configure.m4])

AC_DEFINE_UNQUOTED([uct_MODULES], ["${uct_modules}"], [UCT loadable modules])

AC_CONFIG_FILES([src/uct/Makefile
                 src/uct/ucx-uct.pc])

#
# TCP flags
#
AC_CHECK_DECLS([IPPROTO_TCP, SOL_SOCKET, SO_KEEPALIVE,
                TCP_KEEPCNT, TCP_KEEPIDLE, TCP_KEEPINTVL],
               [],
               [tcp_keepalive_happy=no],
               [[#include <netinet/tcp.h>]
                [#include <netinet/in.h>]])
AS_IF([test "x$tcp_keepalive_happy" != "xno"],
      [AC_DEFINE([UCT_TCP_EP_KEEPALIVE], 1, [Enable TCP keepalive configuration])]);
