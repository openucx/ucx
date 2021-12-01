#
# Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

fuse3_happy="no"

AC_ARG_WITH([fuse3],
            [AS_HELP_STRING([--with-fuse3=(DIR)],
            [Enable the use of FUSEv3 (default is guess).])],
            [], [with_fuse3=guess])

AC_ARG_WITH([fuse3-static],
            [AS_HELP_STRING([--with-fuse3-static=(DIR)],
            [Static link FUSEv3 library (default is no).])],
            [], [with_fuse3_static=no])

AS_IF([test "x$with_fuse3" != "xno" -o "x$with_fuse3_static" != "xno"],
      [
       AS_IF([test "x$with_fuse3_static" != "xno"],
             [with_fuse3="$with_fuse3_static"])

       save_PKG_CONFIG_PATH="$PKG_CONFIG_PATH"

       AS_IF([test "x$with_fuse3" != "xguess" \
                -a "x$with_fuse3" != "xyes" \
                -a "x$with_fuse3" != "x"],
             [AS_IF([test -d "$with_fuse3/lib64"],
                    [libsuff="64"], [libsuff=""])
              ucx_fuse3_pkgconfig="$with_fuse3/lib$libsuff/pkgconfig"
              export PKG_CONFIG_PATH="$ucx_fuse3_pkgconfig:$PKG_CONFIG_PATH"])

       AS_IF([test "x$with_fuse3_static" != "xno"],
             [PKG_CHECK_MODULES_STATIC([FUSE3], [fuse3],
                                       [fuse3_happy="yes"], [fuse3_happy="no"])])

       AS_IF([test "x$fuse3_happy" = "xyes"],
             [FUSE3_LIBS=${FUSE3_LIBS//"-lfuse3"/"-Wl,-Bstatic,-lfuse3,-Bdynamic"}],
             [PKG_CHECK_MODULES([FUSE3], [fuse3],
                                [fuse3_happy="yes"], [fuse3_happy="no"])])

       save_CFLAGS="$CFLAGS"
       save_LIBS="$LIBS"

       CFLAGS="$FUSE3_CFLAGS $CFLAGS"
       LIBS="$FUSE3_LIBS $LIBS"

       AS_IF([test "x$fuse3_happy" = "xyes"],
             [AC_CHECK_DECLS([fuse_open_channel, fuse_mount, fuse_unmount],
                             [AC_SUBST([FUSE3_CFLAGS], [$FUSE3_CFLAGS])
                              AC_DEFINE([FUSE_USE_VERSION], 30, [Fuse API version])],
                             [fuse3_happy="no"],
                             [[#define FUSE_USE_VERSION 30
                               #include <fuse.h>]])])

       AS_IF([test "x$fuse3_happy" = "xyes"],
             [AC_CHECK_FUNCS([fuse_open_channel fuse_mount fuse_unmount],
                             [AC_SUBST([FUSE3_LIBS], ["$FUSE3_LIBS"])],
                             [fuse3_happy="no"])])

       AS_IF([test "x$fuse3_happy" != "xyes" -a "x$with_fuse3" != "xguess"],
             [AC_MSG_ERROR([FUSEv3 requested but could not be found])])

       CFLAGS="$save_CFLAGS"
       LIBS="$save_LIBS"
       export PKG_CONFIG_PATH="$save_PKG_CONFIG_PATH"
    ],
    [AC_MSG_WARN([FUSEv3 was explicitly disabled])]
)

AM_CONDITIONAL([HAVE_FUSE3], [test "x$fuse3_happy" != xno])
vfs_enable=$fuse3_happy
