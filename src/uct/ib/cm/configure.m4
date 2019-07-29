#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
# Copyright (C) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
# Copyright (C) The University of Tennessee and the University of Tennessee Research Foundation. 2016. ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

#
# CM (IB connection manager) Support
#
cm_happy="no"

AC_ARG_WITH([cm],
            [AC_HELP_STRING([--with-cm], [Compile with IB Connection Manager support])],
            [],
            [with_cm=guess])

AS_IF([test "x$with_cm" != xno],
      [save_LIBS="$LIBS"
       AC_CHECK_LIB([ibcm], [ib_cm_send_req],
                    [AC_SUBST(IBCM_LIBS, [-libcm])
                     uct_ib_modules="${uct_ib_modules}:cm"
                     cm_happy="yes"],
                    [AS_IF([test "x$with_cm" = xyes],
                           [AC_MSG_ERROR([CM requested but lib ibcm not found])],
                           [AC_MSG_WARN([CM support not found, skipping])]
                           )
                    ])
       LIBS="$save_LIBS"])

AM_CONDITIONAL([HAVE_TL_CM], [test "x$cm_happy" != xno])
AC_CONFIG_FILES([src/uct/ib/cm/Makefile])
