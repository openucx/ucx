#
# Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

#
# Check for RDMACM support
#
rdmacm_happy="no"
AC_ARG_WITH([rdmacm],
           [AS_HELP_STRING([--with-rdmacm=(DIR)], [Enable the use of RDMACM (default is guess).])],
           [], [with_rdmacm=guess])

AS_IF([test "x$with_rdmacm" != xno],
      [AS_IF([test "x$with_rdmacm" == xguess -o "x$with_rdmacm" == xyes -o "x$with_rdmacm" == x],
             [ucx_check_rdmacm_dir=/usr],
             [ucx_check_rdmacm_dir=$with_rdmacm])

       AS_IF([test -d "$ucx_check_rdmacm_dir/lib64"],[libsuff="64"],[libsuff=""])
       save_LDFLAGS="$LDFLAGS"
       save_CPPFLAGS="$CPPFLAGS"

       AS_IF([test "$ucx_check_rdmacm_dir" != /usr],
             [
             LDFLAGS="-L$ucx_check_rdmacm_dir/lib$libsuff $LDFLAGS"
             CPPFLAGS="-I$ucx_check_rdmacm_dir/include $CPPFLAGS"])

       AC_CHECK_HEADER([$ucx_check_rdmacm_dir/include/rdma/rdma_cma.h],
                       [
                       AC_CHECK_LIB([rdmacm], [rdma_create_id],
                                     [uct_modules+=":rdmacm"
                                      rdmacm_happy="yes"
                                      AS_IF([test "$ucx_check_rdmacm_dir" != /usr],
                                            [
                                            AC_SUBST(RDMACM_CPPFLAGS, ["-I$ucx_check_rdmacm_dir/include"])
                                            AC_SUBST(RDMACM_LDFLAGS,  ["-L$ucx_check_rdmacm_dir/lib$libsuff"])])
                                      AC_SUBST(RDMACM_LIBS,     [-lrdmacm])
                                      # QP less support
                                      AC_CHECK_DECLS(rdma_establish,
                                                     [AC_DEFINE([HAVE_RDMACM_QP_LESS], 1, [RDMA CM QP less support])],
                                                     [],
                                                     [#include <$ucx_check_rdmacm_dir/include/rdma/rdma_cma.h>])
                                     ],
                                     [AC_MSG_WARN([RDMACM requested but librdmacm is not found])
                                      AC_MSG_ERROR([Please install librdmacm and librdmacm-devel or disable rdmacm support])
                                     ])
                       ],
                       [
                       AS_IF([test "x$with_rdmacm" != xguess],
                             [AC_MSG_ERROR([RDMACM requested but required file (rdma/rdma_cma.h) could not be found in $ucx_check_rdmacm_dir])],
                             [AC_MSG_WARN([RDMACM requested but required file (rdma/rdma_cma.h) could not be found in $ucx_check_rdmacm_dir])])
                       ])

       LDFLAGS="$save_LDFLAGS"
       CPPFLAGS="$save_CPPFLAGS"
      ]
)

AM_CONDITIONAL([HAVE_RDMACM], [test "x$rdmacm_happy" != xno])
AC_CONFIG_FILES([src/uct/ib/rdmacm/Makefile])
