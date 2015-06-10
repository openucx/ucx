#
# Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
# $COPYRIGHT$
# $HEADER$
#

cray_ugni_supported=no

AC_ARG_WITH([ugni],
        [AC_HELP_STRING([--with-ugni(=DIR)],
            [Build Cray UGNI support, adding DIR/include, DIR/lib, and DIR/lib64 to the search path for headers and libraries])],
        [with_ugni=forced],
        [with_ugni=yes])

#
# Temporarily disable ugni, until it's adapted to PD API change
# #
#AS_IF([test "x$with_ugni" != "xno"], 
#        [PKG_CHECK_MODULES([CRAY_UGNI], [cray-ugni cray-pmi], 
#                           [transports="${transports},cray-ugni"
#                           cray_ugni_supported=yes],
#                           [AS_IF([test "x$with_ugni" == "xforced"],
#                                  [AC_MSG_WARN([UGNI support was requested but cray-ugni and cray-pmi packages can't be found])
#                                   AC_MSG_ERROR([Cannot continue])],[])]
#                           )])

#
# For automake
#
AM_CONDITIONAL([HAVE_CRAY_UGNI], [test "x$cray_ugni_supported" == xyes])
