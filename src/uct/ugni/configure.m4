#
# Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
# Copyright (C) ARM Ltd. 2020.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

cray_ugni_supported=no

AC_ARG_WITH([ugni],
            [AC_HELP_STRING([--with-ugni(=DIR)], [Build Cray UGNI support])], [], [])

AS_IF([test "x$with_ugni" != "xno"],
      [AC_MSG_CHECKING([cray-ugni])
       AS_IF([$PKG_CONFIG --exists cray-ugni cray-pmi],
             [AC_MSG_RESULT([yes])
              AC_SUBST([CRAY_UGNI_CFLAGS], [`$PKG_CONFIG --cflags cray-ugni cray-pmi`])
              AC_SUBST([CRAY_UGNI_LIBS],   [`$PKG_CONFIG --libs   cray-ugni cray-pmi`])
              uct_modules="${uct_modules}:ugni"
              cray_ugni_supported=yes
              AC_DEFINE([HAVE_TL_UGNI], [1], [Defined if UGNI transport exists])
             ],
             [AC_MSG_RESULT([no])
              AS_IF([test "x$with_ugni" != "x"],
                    [AC_MSG_ERROR([UGNI support was requested but cray-ugni and cray-pmi packages cannot be found])])
             ])])


AM_CONDITIONAL([HAVE_CRAY_UGNI], [test "x$cray_ugni_supported" = xyes])
AC_CONFIG_FILES([src/uct/ugni/Makefile
                 src/uct/ugni/ucx-ugni.pc])
