#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

AS_CASE([$host_os],
    [linux*], [
        SAVE_LDFLAGS="$LDFLAGS"

        LDFLAGS="$LDFLAGS -Xlinker -z -Xlinker interpose -Xlinker --no-as-needed"

        AC_LINK_IFELSE([AC_LANG_PROGRAM([[#include <stdio.h>]])],[
            AC_SUBST([UCM_MODULE_LDFLAGS],
                     ["-Xlinker -z -Xlinker interpose -Xlinker --no-as-needed"])

        ],[
            AC_MSG_ERROR([Required -Xlinker option does not work])
        ])

        LDFLAGS="$SAVE_LDFLAGS"
    ],
    [AC_MSG_ERROR([UCM_MODULE_LDFLAGS=PORT ME])]
)

ucm_modules=""
m4_include([src/ucm/cuda/configure.m4])
m4_include([src/ucm/rocm/configure.m4])
AC_DEFINE_UNQUOTED([ucm_MODULES], ["${ucm_modules}"], [UCM loadable modules])

AC_CONFIG_FILES([src/ucm/Makefile])
