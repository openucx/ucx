#
# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

AC_ARG_WITH([nvml],
            [AS_HELP_STRING([--with-nvml=(DIR)],
            [Enable the use of NVIDIA management library (NVML) (default is guess).])],
            [], [with_nvml=guess])

AS_IF([test "x$with_nvml" != xno],
      [
       NVML_CHECK_CFLAGS=""
       NVML_CHECK_LIBS="-lnvidia-ml"
       AS_IF([test "x$with_nvml" = "xguess" -o "x$with_nvml" = "xyes"],
             [NVML_CHECK_CFLAGS=""
              NVML_CHECK_CPPFLAGS=""
              NVML_CHECK_LIBS=""
              NVML_CHECK_LDFLAGS=""],
             [NVML_CHECK_CFLAGS="-I${with_nvml}/include"
              NVML_CHECK_CPPLAGS="-I${with_nvml}/include"
              NVML_CHECK_LIBS="-lnvidia-ml"
              NVML_CHECK_LDFLAGS="-L${with_nvml}/lib -L${with_nvml}/lib64"])

       save_CFLAGS="$CFLAGS"
       save_CPPLAGS="$CPPFLAGS"
       save_LDFLAGS="$LDFLAGS"
       save_LIBS="$LIBS"

       CFLAGS="$NVML_CHECK_CFLAGS $CFLAGS"
       CPPFLAGS="$NVML_CHECK_CFLAGS $CPPFLAGS"
       LDFLAGS="$NVML_CHECK_LDFLAGS $LDFLAGS"
       LIBS="$NVML_CHECK_LIBS $LIBS"

       nvml_happy="yes"
       AC_CHECK_DECLS([nvmlInit],
                      [], [nvml_happy="no"],
                      [[#include <nvml.h>]])

       # Try to link a simple program using nvmlInit/nvmlShutdown 
       AC_MSG_CHECKING([nvmlInit])
       AC_LINK_IFELSE([AC_LANG_SOURCE([[
                #include <nvml.h>
                int main(int argc, char** argv) {
                    nvmlInit();
                    nvmlShutdown();
                    return 0;
                } ]])],
                [AC_MSG_RESULT([yes])],
                [AC_MSG_RESULT([no])
                 nvml_happy="no"])

       AS_IF([test "x$nvml_happy" = "xyes"],
             [AC_SUBST([NVML_CFLAGS], [${NVML_CHECK_CFLAGS}])
              AC_SUBST([NVML_LIBS], [${NVML_CHECK_LIBS}])
              AC_SUBST([NVML_LDFLAGS], [${NVML_CHECK_LDFLAGS}])],
             [AS_IF([test "x$with_nvml" != "xguess"],
                    [AC_MSG_ERROR([nvml requested but could not be found])])])

       CFLAGS="$save_CFLAGS"
       LDFLAGS="$save_LDFLAGS"
    ],
    [nvml_happy="no"
     AC_MSG_WARN([nvml was explicitly disabled])]
)

AM_CONDITIONAL([HAVE_NVML], [test "x$nvml_happy" = xyes])
AC_CONFIG_FILES([src/ucs/sys/topo/nvml/Makefile])

AS_IF([test "x$nvml_happy" = "xyes"], [ucs_modules="${ucs_modules}:nvml"])
