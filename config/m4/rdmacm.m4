#
# Check for RDMACM support
#
rdmacm_happy="no"
AC_ARG_WITH([rdmacm],
           [AS_HELP_STRING([--with-rdmacm=(DIR)], [Enable the use of RDMACM (default is guess).])],
           [], [with_rdmacm=yes])

AS_IF([test "x$with_rdmacm" != xno],
      [AS_IF([test "x$with_rdmacm" == xyes],
             [with_rdmacm=/usr])

       AS_IF([test -d "$with_rdmacm/lib64"],[libsuff="64"],[libsuff=""])
       save_LDFLAGS="$LDFLAGS"
       save_CFLAGS="$CFLAGS"
       save_CPPFLAGS="$CPPFLAGS"

       LDFLAGS="-L$with_rdmacm/lib$libsuff $LDFLAGS"
       CFLAGS="-I$with_rdmacm/include $CFLAGS"
       CPPFLAGS="-I$with_rdmacm/include $CPPFLAGS"

       AC_CHECK_HEADER([$with_rdmacm/include/rdma/rdma_cma.h],
                       [
                       AC_CHECK_LIB([rdmacm], [rdma_create_id],
                                     [transports="${transports},rdmacm"
                                      rdmacm_happy="yes"
                                      AC_SUBST(RDMACM_CPPFLAGS, ["-I$with_rdmacm/include"])
                                      AC_SUBST(RDMACM_CFLAGS,   ["-I$with_rdmacm/include"])
                                      AC_SUBST(RDMACM_LDFLAGS,  ["-L$with_rdmacm/lib$libsuff -lrdmacm"])
                                      AC_SUBST(RDMACM_LIBS,     [-lrdmacm])
                                     ], 
                                     [AC_MSG_WARN([RDMACM requested but librdmacm is not found])
                                      AC_MSG_ERROR([Please install librdmacm and librdmacm-devel or disable rdmacm support])
                                     ])
                       ],
                       [AC_MSG_ERROR([RDMACM requested but required file (rdma/rdma_cma.h) could not be found in $with_rdmacm])])

       LDFLAGS="$save_LDFLAGS"
       CFLAGS="$save_CFLAGS"
       CPPFLAGS="$save_CPPFLAGS"
      ]
)

AM_CONDITIONAL([HAVE_RDMACM], [test "x$rdmacm_happy" != xno])
