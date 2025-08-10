#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2017. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

AC_DEFUN([UCX_CHECK_CUDA],[

AS_IF([test "x$cuda_checked" != "xyes"],
   [
    AC_ARG_WITH([cuda],
                [AS_HELP_STRING([--with-cuda=(DIR)], [Enable the use of CUDA (default is guess).])],
                [], [with_cuda=guess])

    AS_IF([test "x$with_cuda" = "xno"],
        [
         cuda_happy=no
         have_cuda_static=no
         NVCC=""
        ],
        [
         save_CPPFLAGS="$CPPFLAGS"
         save_LDFLAGS="$LDFLAGS"
         save_LIBS="$LIBS"

         CUDA_CPPFLAGS=""
         CUDA_LDFLAGS=""
         CUDA_LIBS=""
         CUDART_LIBS=""
         CUDART_STATIC_LIBS=""
         NVML_LIBS=""
         CUDA_BIN_PATH=""

         AS_IF([test ! -z "$with_cuda" -a "x$with_cuda" != "xyes" -a "x$with_cuda" != "xguess"],
               [ucx_check_cuda_dir="$with_cuda"
                AS_IF([test -d "$with_cuda/lib64"], [libsuff="64"], [libsuff=""])
                ucx_check_cuda_libdir="$with_cuda/lib$libsuff"
                CUDA_CPPFLAGS="-I$with_cuda/include"
                CUDA_LDFLAGS="-L$ucx_check_cuda_libdir -L$ucx_check_cuda_libdir/stubs"
                CUDA_BIN_PATH="$with_cuda/bin"])

         CPPFLAGS="$CPPFLAGS $CUDA_CPPFLAGS"
         LDFLAGS="$LDFLAGS $CUDA_LDFLAGS"

         # Check cuda header files
         AC_CHECK_HEADERS([cuda.h cuda_runtime.h],
                          [cuda_happy="yes"], [cuda_happy="no"])

         # Check cuda libraries
         AS_IF([test "x$cuda_happy" = "xyes"],
               [AC_CHECK_LIB([cuda], [cuDeviceGetUuid],
                             [CUDA_LIBS="$CUDA_LIBS -lcuda"], [cuda_happy="no"])])
         AS_IF([test "x$cuda_happy" = "xyes"],
               [AC_CHECK_LIB([cudart], [cudaGetDeviceCount],
                             [CUDART_LIBS="$CUDART_LIBS -lcudart"], [cuda_happy="no"])])
         # Check optional cuda library members
         AS_IF([test "x$cuda_happy" = "xyes"],
               [AC_CHECK_LIB([cuda], [cuMemRetainAllocationHandle],
                             [AC_DEFINE([HAVE_CUMEMRETAINALLOCATIONHANDLE], [1],
                                        [Enable cuMemRetainAllocationHandle() usage])])
                AC_CHECK_DECLS([CU_MEM_LOCATION_TYPE_HOST],
                               [], [], [[#include <cuda.h>]])])

         # Check nvml header files
         AS_IF([test "x$cuda_happy" = "xyes"],
               [AC_CHECK_HEADERS([nvml.h],
                                 [cuda_happy="yes"],
                                 [AS_IF([test "x$with_cuda" != "xguess"],
                                        [AC_MSG_ERROR([nvml header not found. Install appropriate cuda-nvml-devel package])])
                                  cuda_happy="no"])])

         # Check nvml library
         AS_IF([test "x$cuda_happy" = "xyes"],
               [AC_CHECK_LIB([nvidia-ml], [nvmlInit],
                             [NVML_LIBS="$NVML_LIBS -lnvidia-ml"],
                             [AS_IF([test "x$with_cuda" != "xguess"],
                                    [AC_MSG_ERROR([libnvidia-ml not found. Install appropriate nvidia-driver package])])
                              cuda_happy="no"])])

         # Check for nvmlDeviceGetGpuFabricInfoV
         AC_CHECK_DECLS([nvmlDeviceGetGpuFabricInfoV],
                        [AC_DEFINE([HAVE_NVML_FABRIC_INFO], 1, [Enable NVML GPU fabric info support])],
                        [AC_MSG_NOTICE([nvmlDeviceGetGpuFabricInfoV function not found in libnvidia-ml. MNNVL support will be disabled.])],
                        [[#include <nvml.h>]])


         # Check for cuda static library
         have_cuda_static="no"
         AS_IF([test "x$cuda_happy" = "xyes"],
               [AC_CHECK_LIB([cudart_static], [cudaGetDeviceCount],
                             [CUDART_STATIC_LIBS="$CUDART_STATIC_LIBS -lcudart_static -lrt -ldl -lpthread"
                              have_cuda_static="yes"],
                             [], [-ldl -lrt -lpthread])])

         AC_CHECK_DECLS([CU_MEM_HANDLE_TYPE_FABRIC],
                        [AC_DEFINE([HAVE_CUDA_FABRIC], 1, [Enable CUDA fabric handle support])],
                        [], [[#include <cuda.h>]])

         # Check NVCC exists and able to compile
         nvcc_happy="no"
         AC_PATH_PROGS(NVCC, nvcc, "", $CUDA_BIN_PATH:$PATH)
         AS_IF([test "x$NVCC" != "x"],
               [AC_LANG_PUSH([C])
                AC_LANG_CONFTEST([AC_LANG_SOURCE([[#include <cuda_runtime.h>]])])
                mv conftest.c conftest.cu
                AC_MSG_CHECKING([$NVCC can compile])
                AS_IF([$NVCC -c conftest.cu 2>&AS_MESSAGE_LOG_FD],
                  [AC_MSG_RESULT([yes])
                   nvcc_happy="yes"],
                  [AC_MSG_RESULT([no])
                   cat conftest.cu >&AS_MESSAGE_LOG_FD])
                rm conftest.cu
                AC_LANG_POP
                ])

         CPPFLAGS="$save_CPPFLAGS"
         LDFLAGS="$save_LDFLAGS"
         LIBS="$save_LIBS"

         AS_IF([test "x$cuda_happy" = "xyes"],
               [AC_SUBST([CUDA_CPPFLAGS], ["$CUDA_CPPFLAGS"])
                AC_SUBST([CUDA_LDFLAGS], ["$CUDA_LDFLAGS"])
                AC_SUBST([CUDA_LIBS], ["$CUDA_LIBS"])
                AC_SUBST([CUDART_LIBS], ["$CUDART_LIBS"])
                AC_SUBST([NVML_LIBS], ["$NVML_LIBS"])
                AC_SUBST([CUDART_STATIC_LIBS], ["$CUDART_STATIC_LIBS"])
                AC_DEFINE([HAVE_CUDA], 1, [Enable CUDA support])],
               [AS_IF([test "x$with_cuda" != "xguess"],
                      [AC_MSG_ERROR([CUDA support is requested but cuda packages cannot be found])],
                      [AC_MSG_WARN([CUDA not found])])])
        ]) # "x$with_cuda" = "xno"

        cuda_checked=yes
        AM_CONDITIONAL([HAVE_CUDA], [test "x$cuda_happy" != xno])
        AM_CONDITIONAL([HAVE_CUDA_STATIC], [test "X$have_cuda_static" = "Xyes"])
        AM_CONDITIONAL([HAVE_NVCC], [test "x$nvcc_happy" != xno])

   ]) # "x$cuda_checked" != "xyes"

]) # UCX_CHECK_CUDA
