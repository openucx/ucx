#
# Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

fuse3_happy="no"

AC_ARG_WITH([fuse3],
            [AS_HELP_STRING([--with-fuse3=(DIR)],
            [Enable the use of FUSEv3 (default is guess).])],
            [], [with_fuse3=guess])

AS_IF([test "x$with_fuse3" != xno],
      [
       AS_IF([test "x$with_fuse3" = "xguess" \
                -o "x$with_fuse3" = "xyes" \
                -o "x$with_fuse3" = "x"],
             [FUSE3_CPPFLAGS=$(pkg-config --cflags fuse3)
              FUSE3_LIBS=$(pkg-config --libs fuse3)],
             [FUSE3_CPPFLAGS="-I${with_fuse3}/include/fuse3"
              FUSE3_LIBS="-L${with_fuse3}/lib -L${with_fuse3}/lib64"])

       save_CPPFLAGS="$CPPFLAGS"
       save_LDFLAGS="$LDFLAGS"

       CPPFLAGS="$FUSE3_CPPFLAGS $CPPFLAGS"
       LDFLAGS="$FUSE3_LIBS $LDFLAGS"

       fuse3_happy="yes"
       AC_CHECK_DECLS([fuse_open_channel, fuse_mount, fuse_unmount],
                      [AC_SUBST([FUSE3_CPPFLAGS], [$FUSE3_CPPFLAGS])
                       AC_DEFINE([FUSE_USE_VERSION], 30, [Fuse API version])],
                      [fuse3_happy="no"],
                      [[#define FUSE_USE_VERSION 30
                        #include <fuse.h>]])

       AC_CHECK_FUNCS([fuse_open_channel fuse_mount fuse_unmount],
                      [AC_SUBST([FUSE3_LIBS], [$FUSE3_LIBS])],
                      [fuse3_happy="no"])

       AS_IF([test "x$fuse3_happy" != "xyes" -a "x$with_fuse3" != "xguess"],
             [AC_MSG_ERROR([FUSEv3 requested but could not be found])])

       CPPFLAGS="$save_CPPFLAGS"
       LDFLAGS="$save_LDFLAGS"
    ],
    [AC_MSG_WARN([FUSEv3 was explicitly disabled])]
)

AM_CONDITIONAL([HAVE_FUSE3], [test "x$fuse3_happy" != xno])
vfs_enable=$fuse3_happy
