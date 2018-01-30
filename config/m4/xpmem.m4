#
# Check for XPMEM support
#
xpmem_happy="no"
AC_ARG_WITH([xpmem],
           [AS_HELP_STRING([--with-xpmem=(DIR)], [Enable the use of XPMEM (default is guess).])],
           [], [with_xpmem=guess])

AS_IF([test "x$with_xpmem" != xno],
      [AS_IF([test ! -d $with_xpmem],
             [AC_MSG_NOTICE([XPMEM - failed to open the requested location ($with_xpmem), guessing ...])
             # Check for a Cray module
             PKG_CHECK_MODULES([CRAY_XPMEM], [cray-xpmem],
                               [transports="${transports},xpmem"
                                xpmem_happy=yes
                                AC_SUBST(XPMEM_CPPFLAGS,  "$CRAY_XPMEM_CFLAGS")
                                AC_SUBST(XPMEM_LDFLAGS,   "$CRAY_XPMEM_LIBS")
                                AC_DEFINE([HAVE_XPMEM], [1], [Enable the use of XPMEM])],
                                # If Cray module failed try to search
                                [with_xpmem=$(find /opt/xpmem /usr/local/include /usr/local/xpmem -name xpmem.h 2>/dev/null |
                                              xargs dirname | head -1 | sed -e s,/include,,g)])],
                                [])], [])

# Verify XPMEM header file
AS_IF([test -d $with_xpmem],
      [AC_CHECK_HEADER([$with_xpmem/include/xpmem.h],
                       [AC_SUBST(XPMEM_CPPFLAGS,  "-I$with_xpmem/include")
                       AC_SUBST(XPMEM_LDFLAGS,   "-L$with_xpmem/lib -lxpmem")
                       AC_DEFINE([HAVE_XPMEM], [1], [Enable the use of XPMEM])
                       transports="${transports},xpmem"
                       xpmem_happy="yes"],
                       [AC_MSG_WARN([XPMEM header was not found])
                       AC_DEFINE([HAVE_XPMEM], [0], [Disable the use of XPMEM])])],
      [AC_MSG_WARN([XPMEM was disabled])
       AC_DEFINE([HAVE_XPMEM], [0], [Disable the use of XPMEM])])

AM_CONDITIONAL([HAVE_XPMEM], [test "x$xpmem_happy" != xno])
