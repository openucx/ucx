#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
# Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

xpmem_happy="no"
AC_ARG_WITH([xpmem],
            [AS_HELP_STRING([--with-xpmem=(DIR)], [Enable the use of XPMEM (default is guess).])],
            [], [with_xpmem=guess])

AS_IF([test "x$with_xpmem" != "xno"],
      [AS_IF([test ! -d "$with_xpmem"],
             [
              AC_MSG_NOTICE([XPMEM - failed to open the requested location ($with_xpmem), guessing ...])
              PKG_CHECK_MODULES(
                  [CRAY_XPMEM], [cray-xpmem],
                  [
                   xpmem_happy=yes
                   AC_SUBST(XPMEM_CPPFLAGS, "$CRAY_XPMEM_CFLAGS")
                   AC_SUBST(XPMEM_LDFLAGS,  "$CRAY_XPMEM_LIBS")
                  ],
                  [
                   # If cray-xpmem module not found in pkg-config, try to search
                   xpmem_header=$(find /opt/xpmem /usr/local/include /usr/local/xpmem -name xpmem.h 2>/dev/null|head -1)
                   AS_IF([test -f "$xpmem_header"],
                         [with_xpmem=$(dirname $xpmem_header | head -1 | sed -e s,/include,,g)])
                  ])
              ])
       ])

# Verify XPMEM header file
AS_IF([test "x$xpmem_happy" == "xno" -a -d "$with_xpmem"],
      [AC_CHECK_HEADER([$with_xpmem/include/xpmem.h],
                       [AC_SUBST(XPMEM_CPPFLAGS, "-I$with_xpmem/include")
                        AC_SUBST(XPMEM_LDFLAGS,  "-L$with_xpmem/lib -lxpmem")
                        xpmem_happy="yes"],
                       [AC_MSG_WARN([cray-xpmem header was not found in $with_xpmem])])
       ])

AS_IF([test "x$xpmem_happy" == "xyes"], [uct_modules+=":xpmem"])
AM_CONDITIONAL([HAVE_XPMEM], [test "x$xpmem_happy" != "xno"])
AC_CONFIG_FILES([src/uct/sm/mm/xpmem/Makefile])
