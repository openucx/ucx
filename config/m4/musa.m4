#
# Copyright (C) MThreads 2026. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

AC_DEFUN([UCX_CHECK_MUSA],[
    AS_IF([test "x$musa_checked" != "xyes"], [
        AC_ARG_WITH([musa],
                    [AS_HELP_STRING([--with-musa=(DIR)],
                                    [Enable MUSA 5.2.0 support (default is guess).])],
                    [],
                    [with_musa=guess])

        musa_happy=no
        MUSA_CPPFLAGS=""
        MUSA_LDFLAGS=""
        MUSA_LIBS=""
        MUSA_ROOT=""

        AS_IF([test "x$with_musa" = "xno"], [
            AC_MSG_WARN([MUSA support was explicitly disabled])
        ], [
            AS_IF([test "x$with_musa" = "xyes" ||
                   test "x$with_musa" = "xguess"], [
                AS_IF([test -d /usr/local/musa],
                      [MUSA_ROOT=/usr/local/musa])
            ], [
                MUSA_ROOT="$with_musa"
            ])

            AS_IF([test -n "$MUSA_ROOT"], [
                MUSA_CPPFLAGS="-I$MUSA_ROOT/include"
                MUSA_LDFLAGS="-L$MUSA_ROOT/lib -L$MUSA_ROOT/lib64"
                MUSA_LIBS="-lmusart -lmusa"

                musa_happy=yes
                save_CPPFLAGS="$CPPFLAGS"
                save_LDFLAGS="$LDFLAGS"
                save_LIBS="$LIBS"

                CPPFLAGS="$MUSA_CPPFLAGS $CPPFLAGS"
                LDFLAGS="$MUSA_LDFLAGS $LDFLAGS"
                LIBS="$MUSA_LIBS $LIBS"

                AC_CHECK_HEADERS([musa_runtime_api.h],
                                 [],
                                 [musa_happy=no])
                AC_CHECK_HEADERS([musa.h],
                                 [],
                                 [musa_happy=no])

                AS_IF([test "x$musa_happy" = "xyes"], [
                    AC_CHECK_LIB([musart], [musaGetDeviceCount],
                                 [],
                                 [musa_happy=no])
                ])
                AS_IF([test "x$musa_happy" = "xyes"], [
                    AC_CHECK_LIB([musart], [musaMalloc],
                                 [],
                                 [musa_happy=no])
                    AC_CHECK_LIB([musart], [musaFree],
                                 [],
                                 [musa_happy=no])
                    AC_CHECK_LIB([musart], [musaMemcpy],
                                 [],
                                 [musa_happy=no])
                    AC_CHECK_LIB([musart], [musaHostRegister],
                                 [],
                                 [musa_happy=no])
                    AC_CHECK_LIB([musart], [musaHostUnregister],
                                 [],
                                 [musa_happy=no])
                    AC_CHECK_LIB([musart], [musaDeviceGetPCIBusId],
                                 [],
                                 [musa_happy=no])
                ])
                AS_IF([test "x$musa_happy" = "xyes"], [
                    AC_CHECK_LIB([musa], [muMemGetAddressRange],
                                 [],
                                 [musa_happy=no])
                ])
                AS_IF([test "x$musa_happy" = "xyes"], [
                    AC_CHECK_FUNCS([musaGetDeviceCount
                                   musaGetDevice
                                   musaSetDevice
                                   musaPointerGetAttributes
                                   musaGetErrorString])
                ])

                AS_IF([test "x$musa_happy" = "xyes"], [
                    AC_LINK_IFELSE(
                        [AC_LANG_PROGRAM(
                            [[#include <stddef.h>
                              #include <musa.h>
                              #include <musa_runtime_api.h>]],
                            [[int count = 0;
                              int device = 0;
                              void *address = 0;
                              MUdeviceptr base = 0;
                              size_t size = 0;
                              musaGetDeviceCount(&count);
                              musaGetDevice(&device);
                              musaSetDevice(device);
                              musaPointerGetAttributes(NULL, address);
                              musaGetErrorString(musaSuccess);
                              muMemGetAddressRange(&base, &size,
                                                   (MUdeviceptr)address);
                              return count < 0;]])],
                        [],
                        [musa_happy=no])
                ])

                CPPFLAGS="$save_CPPFLAGS"
                LDFLAGS="$save_LDFLAGS"
                LIBS="$save_LIBS"
            ])

            AS_IF([test "x$musa_happy" = "xyes"], [
                AC_DEFINE([HAVE_MUSA], [1], [Enable MUSA support])
                AC_SUBST([MUSA_CPPFLAGS])
                AC_SUBST([MUSA_LDFLAGS])
                AC_SUBST([MUSA_LIBS])
                AC_SUBST([MUSA_ROOT])
            ], [
                AS_IF([test "x$with_musa" != "xguess"], [
                    AC_MSG_ERROR([MUSA support was requested but MUSA 5.2.0 headers and libraries were not found])
                ], [
                    AC_MSG_WARN([MUSA 5.2.0 not found])
                ])
            ])
        ])

        musa_checked=yes
        AM_CONDITIONAL([HAVE_MUSA], [test "x$musa_happy" = "xyes"])
    ])
])
