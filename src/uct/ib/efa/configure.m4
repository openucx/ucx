#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# See file LICENSE for terms.
#

#
# EFA DV Support
#
AC_ARG_WITH([efa-dv],
            [AC_HELP_STRING([--with-efa-dv=(DIR)], [Compile with EFA device support])],
            [], [with_efa_dv=/usr])

#
# SRD Support
#
AC_ARG_WITH([srd],
            [AC_HELP_STRING([--with-srd],
                            [Compile with EFA Scalable Reliable Datagram support])],
            [],
            [with_srd=yes])


AS_IF([test "x$with_ib" = "xno"], [with_efa_dv=no])

AS_IF([test "x$with_efa_dv" = "xyes"], [with_efa_dv=/usr])

AS_IF([test -d "$with_efa_dv"],
      [str="with EFA support from $with_efa_dv"],
      [with_efa_dv=no; str="without EFA support"])



# Check the EFA header
AS_IF([test "x$with_efa_dv" != xno],
      [
       save_CFLAGS="$CFLAGS"
       save_CPPFLAGS="$CPPFLAGS"
       save_LDFLAGS="$LDFLAGS"
       save_LIBS="$LIBS"

       AS_IF([test -d "$with_efa_dv/lib64"],[libsuff="64"],[libsuff=""])
       AS_IF([test "x$with_efa_dv" = "x/usr"], [],
             [efa_incl_dir="-I$with_efa_dv/include"
              efa_libs_dir="-L$with_efa_dv/lib$libsuff"])

       CFLAGS="$efa_incl_dir $CFLAGS"
       CPPFLAGS="$efa_incl_dir $CPPFLAGS"

       AC_CHECK_HEADER([infiniband/efadv.h],
                       [AC_SUBST(EFA_CFLAGS,   ["$efa_incl_dir"])
                        AC_SUBST(EFA_CPPFLAGS, ["$efa_incl_dir"])],
                       [AC_MSG_WARN([EFA header file not found]); with_efa_dv=no])

       CFLAGS="$save_CFLAGS"
       CPPFLAGS="$save_CPPFLAGS"])

# Check EFA libriary
AS_IF([test "x$with_efa_dv" != xno],
      [
       LDFLAGS="$efa_libs_dir $LDFLAGS"

       AC_CHECK_LIB([efa], [efadv_query_device],
                    [AC_SUBST(EFA_LDFLAGS, ["$efa_libs_dir"])
                     AC_SUBST(EFA_LIBS, [-lefa])
                     AC_DEFINE([HAVE_EFA_DV], [1], [EFA device support])],
                    [with_efa_dv=no])

       LDFLAGS="$save_LDFLAGS"])

AS_IF([test "x$with_efa_dv" != xno],
      [
       CFLAGS="$EFA_CFLAGS $CFLAGS"
       CPPFLAGS="$EFA_CPPFLAGS $CPPFLAGS"

       AC_CHECK_DECLS([EFADV_DEVICE_ATTR_CAPS_RDMA_READ],
                      [AC_DEFINE([HAVE_DECL_EFA_DV_RDMA_READ],
                                 [1], [HAVE EFA device with RDMA READ support])],
                      [], [[#include <infiniband/efadv.h>]])

       CFLAGS="$save_CFLAGS"
       CPPFLAGS="$save_CPPFLAGS"])

AS_IF([test "x$with_efa_dv" != xno -a "x$with_srd" != xno],
      [AC_DEFINE([HAVE_TL_SRD], 1, [SRD transport support])],
      [with_srd=no])

# Add EFA to IB modules
AS_IF([test "x$with_efa_dv" != xno], [uct_ib_modules="${uct_ib_modules}:efa"])

#
# For automake
#
AM_CONDITIONAL([HAVE_EFA_DV],  [test "x$with_efa_dv" != xno])
AM_CONDITIONAL([HAVE_TL_SRD],  [test "x$with_srd" != xno])

AC_CONFIG_FILES([src/uct/ib/efa/Makefile])
