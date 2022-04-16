#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#


SAVE_LDFLAGS="$LDFLAGS"

#
# Linux
#
UCM_MODULE_LDFLAGS_TEST="-Xlinker -z -Xlinker interpose -Xlinker --no-as-needed"
LDFLAGS="$SAVE_LDFLAGS $UCM_MODULE_LDFLAGS_TEST"
AC_LINK_IFELSE([AC_LANG_PROGRAM([])],[
    AC_SUBST([UCM_MODULE_LDFLAGS],[$UCM_MODULE_LDFLAGS_TEST])
    ucm_ldflags_happy=yes
],[
    ucm_ldflags_happy=no
])

#
# ERROR
#
AS_IF([test "x$ucm_ldflags_happy" = "xno"],[
    AC_MSG_ERROR([UCM linker flags are not supported])
],[])

LDFLAGS="$SAVE_LDFLAGS"

ucm_modules=""
m4_include([src/ucm/cuda/configure.m4])
m4_include([src/ucm/rocm/configure.m4])
AC_DEFINE_UNQUOTED([ucm_MODULES], ["${ucm_modules}"], [UCM loadable modules])

AC_CONFIG_FILES([src/ucm/Makefile])
