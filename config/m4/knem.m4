#
# Check for KNEM support
#
knem_happy="no"
AC_ARG_WITH([knem],
           [AS_HELP_STRING([--with-knem=(DIR)], [Enable the use of KNEM (default is NO).])],
           [], [with_knem=no])

AS_IF([test "x$with_knem" != xno],
      [AS_IF([test ! -d $with_knem], 
             [
              AC_MSG_NOTICE([KNEM path was not found, guessing ...])
              with_knem=$(${PKG_CONFIG} --variable=prefix knem || find /opt/knem* -name knem_io.h |xargs dirname |sed -e s,/include,,g)
              ],
              [:])
       AC_CHECK_HEADER([$with_knem/include/knem_io.h],
                       [CFLAGS="$CFLAGS -I$with_knem/include"
                        CPPFLAGS="$CPPFLAGS -I$with_knem/include"
                        AC_DEFINE([HAVE_KNEM], [1], [Enable the use of KNEM])
                        transports="${transports},knem"
                        knem_happy="yes"],
                       [AC_MSG_WARN([KNEM requested but not found])
                        AC_DEFINE([HAVE_KNEM], [0], [Disable the use of KNEM])])],
      [AC_MSG_WARN([KNEM was explicitly disabled])
       AC_DEFINE([HAVE_KNEM], [0], [Disable the use of KNEM])]
)

AM_CONDITIONAL([HAVE_KNEM], [test "x$knem_happy" != xno])
