#
# Check for XPMEM support
#
xpmem_happy="no"
AC_ARG_WITH([xpmem],
           [AS_HELP_STRING([--with-xpmem=(DIR)], [Enable the use of XPMEM (default is NO).])],
           [], [with_xpmem=guess])

AS_IF([test "x$with_xpmem" != xno],
      [AS_IF([test ! -d $with_xpmem], 
             [
              AC_MSG_NOTICE([XPMEM path was not found, guessing ...])
              with_xpmem=$(${PKG_CONFIG} --variable=prefix xpmem || find /opt/xpmem* -name xpmem.h |xargs dirname |sed -e s,/include,,g)
              ],
              [:])
       AC_CHECK_HEADER([$with_xpmem/include/xpmem.h],
                       [AC_SUBST(XPMEM_CPPFLAGS,  "-I$with_xpmem/include")
                        AC_SUBST(XPMEM_LDFLAGS,   "-L$with_xpmem/lib -lxpmem")
                        AC_DEFINE([HAVE_XPMEM], [1], [Enable the use of XPMEM])
                        transports="${transports},xpmem"
                        xpmem_happy="yes"],
                       [AC_MSG_WARN([XPMEM requested but not found])
                        AC_DEFINE([HAVE_XPMEM], [0], [Disable the use of XPMEM])])],
      [AC_MSG_WARN([XPMEM was explicitly disabled])
       AC_DEFINE([HAVE_XPMEM], [0], [Disable the use of XPMEM])]
)

AM_CONDITIONAL([HAVE_XPMEM], [test "x$xpmem_happy" != xno])
