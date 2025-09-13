#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

AC_ARG_WITH([doca-gpunetio],
            [AS_HELP_STRING([--with-doca-gpunetio=(DIR)],
                            [Use DOCA gpunetio (Default is guess)])],
            [with_doca_gpunetio=$withval],
            [with_doca_gpunetio=guess])


AS_IF([test "x$cuda_happy" = "xyes"],
      [
       # Default value
       GPUNETIO_CFLAGS=""
       GPUNETIO_LDFLAGS=""
       GPUNETIO_LIBS="-ldoca_gpunetio"
       AS_IF([test "x$with_doca_gpunetio" != "xno"],
             [
              AS_IF([test "x$with_doca_gpunetio" = "xguess"],
                    [
                     AS_IF([$PKG_CONFIG --exists doca-gpunetio],
                           [GPUNETIO_CFLAGS=$(pkg-config --cflags doca-gpunetio)
                            GPUNETIO_LDFLAGS=$(pkg-config --libs-only-L doca-gpunetio)
                            GPUNETIO_LIBS=$(pkg-config --libs-only-l doca-gpunetio)])
                    ],
                    [
                     GPUNETIO_CFLAGS="-I${with_doca_gpunetio}/include"
                     for doca_libdir in lib/x86_64-linux-gnu lib64; do
                         if test -d "${with_doca_gpunetio}/${doca_libdir}"; then
                             GPUNETIO_LDFLAGS="$GPUNETIO_LDFLAGS -L${with_doca_gpunetio}/${doca_libdir} "
                             # Add rpath-link to search for doca_gpunetio dependencies
                             GPUNETIO_LDFLAGS="$GPUNETIO_LDFLAGS -Wl,-rpath-link,${with_doca_gpunetio}/${doca_libdir}"
                         fi
                     done
                     # Add CUDA lib dirs to rpath-link for gpunetio
                     for cuda_libdir in $CUDA_LIB_DIRS; do
                         GPUNETIO_LDFLAGS="$GPUNETIO_LDFLAGS -Wl,-rpath-link,${cuda_libdir}"
                     done
                    ]) # "x$with_doca_gpunetio" != "xguess"
             ]) # "x$with_doca_gpunetio" != "xno"

       save_CPPFLAGS="$CPPFLAGS"
       save_LDFLAGS="$LDFLAGS"
       CPPFLAGS="$CPPFLAGS $CUDA_CFLAGS $GPUNETIO_CFLAGS"
       LDFLAGS="$LDFLAGS $CUDA_LDFLAGS $GPUNETIO_LDFLAGS"

       gpunetio_happy=yes
       AC_CHECK_HEADERS([doca_gpunetio.h], [], [gpunetio_happy=no])
       AC_CHECK_LIB([doca_gpunetio], [doca_gpu_verbs_bridge_export_qp],
                    [true], [gpunetio_happy=no], [$GPUNETIO_LIBS])

       CPPFLAGS="$save_CPPFLAGS"
       LDFLAGS="$save_LDFLAGS"
      ],
      [gpunetio_happy=no])

AS_IF([test "x$gpunetio_happy" = "xyes"],
      [
       uct_ib_mlx5_modules="${uct_ib_mlx5_modules}:gda"
       AC_SUBST(GPUNETIO_CFLAGS)
       AC_SUBST(GPUNETIO_LDFLAGS)
       AC_SUBST(GPUNETIO_LIBS)
      ],
      [
       # gpunetio was requested but not found
       AS_IF([test "x$with_doca_gpunetio" != "xno" -a "x$with_doca_gpunetio" != "xguess"],
             [AC_MSG_ERROR([doca_gpunetio not found])])
      ])

AM_CONDITIONAL([HAVE_GPUNETIO], [test x$gpunetio_happy = xyes])
AC_CONFIG_FILES([src/uct/ib/mlx5/gdaki/Makefile
                 src/uct/ib/mlx5/gdaki/ucx-ib-mlx5-gda.pc])
