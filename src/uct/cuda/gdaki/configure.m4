#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

#
# pkg-config file for DOCA on Ubuntu 22.04 is missing -ldoca_verbs -ldoca_common
# so we need to add them manually using --with-doca-gpunetio arg for configure
AC_ARG_WITH([doca-gpunetio],
            [AS_HELP_STRING([--with-doca-gpunetio],
                            [Use DOCA gpunetio])],
            [with_doca_gpunetio=$withval],
            [with_doca_gpunetio=guess])

# Default value
GPUNETIO_CFLAGS=""
GPUNETIO_LDFLAGS=""
GPUNETIO_LIBS="-ldoca_gpunetio"

AS_IF([test x$with_doca_gpunetio != xno], [
  AS_IF([test x$with_doca_gpunetio = xguess],
    [
      AS_IF([$PKG_CONFIG --exists doca-gpunetio],
            [
              # Guess from pkg-config
              GPUNETIO_CFLAGS=$(pkg-config --cflags doca-gpunetio)
              GPUNETIO_LDFLAGS=$(pkg-config --libs-only-L doca-gpunetio)
              GPUNETIO_LIBS=$(pkg-config --libs-only-l doca-gpunetio)
            ])
    ],
    [
      # User provided path
      GPUNETIO_CFLAGS="-I${with_doca_gpunetio}/include"
      for dir in lib lib64 lib/x86_64-linux-gnu; do
        if test -d "${with_doca_gpunetio}/${dir}"; then
          GPUNETIO_LDFLAGS="$GPUNETIO_LDFLAGS -L${with_doca_gpunetio}/${dir} "
          # Add rpath-link to seach for doca_gpunetio dependencies
          GPUNETIO_LDFLAGS="$GPUNETIO_LDFLAGS -Wl,-rpath-link,${with_doca_gpunetio}/${dir}"
        fi
      done
    ])
])

save_CPPFLAGS="$CPPFLAGS"
save_LDFLAGS="$LDFLAGS"
CPPFLAGS="$CPPFLAGS $GPUNETIO_CFLAGS"
LDFLAGS="$LDFLAGS $GPUNETIO_LDFLAGS"

AC_CHECK_HEADERS([doca_gpunetio.h], [have_gpunetio=yes], [have_gpunetio=no])
AC_CHECK_LIB([doca_gpunetio], [doca_gpu_verbs_bridge_export_qp],
             [true], [have_gpunetio=no], [$GPUNETIO_LIBS])

CPPFLAGS="$save_CPPFLAGS"
LDFLAGS="$save_LDFLAGS"

AS_IF([test x$have_gpunetio = xyes], [
    uct_cuda_modules="${uct_cuda_modules}:gdaki"
    AC_SUBST(GPUNETIO_CFLAGS)
    AC_SUBST(GPUNETIO_LDFLAGS)
    AC_SUBST(GPUNETIO_LIBS)
])
AM_CONDITIONAL([HAVE_GPUNETIO], [test x$have_gpunetio = xyes])

AC_CONFIG_FILES([src/uct/cuda/gdaki/Makefile
                 src/uct/cuda/gdaki/ucx-gdaki.pc])
