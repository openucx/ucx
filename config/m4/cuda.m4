#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2017. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

NVCC_CUDA_MIN_REQUIRED_MAJOR=10
NVCC_CUDA_MIN_REQUIRED_MINOR=2

ARCH9_CODE="-gencode=arch=compute_70,code=sm_70"
ARCH10_CODE="-gencode=arch=compute_75,code=sm_75"
ARCH110_CODE="-gencode=arch=compute_80,code=sm_80"
ARCH111_CODE="-gencode=arch=compute_86,code=sm_86"
ARCH120_CODE="-gencode=arch=compute_90,code=sm_90"
ARCH124_CODE="-gencode=arch=compute_89,code=sm_89"
ARCH128_CODE="-gencode=arch=compute_100,code=sm_100 -gencode=arch=compute_120,code=sm_120"
ARCH130_CODE="-gencode=arch=compute_110,code=sm_110"


ARCH9_PTX="-gencode=arch=compute_70,code=compute_70"
ARCH10_PTX=""
ARCH110_PTX="-gencode=arch=compute_80,code=compute_80"
ARCH111_PTX="-gencode=arch=compute_86,code=compute_86"
ARCH120_PTX="-gencode=arch=compute_90,code=compute_90"
ARCH124_PTX="-gencode=arch=compute_90,code=compute_90"
ARCH128_PTX="-gencode=arch=compute_120,code=compute_120"
ARCH130_PTX="-gencode=arch=compute_120,code=compute_120"

# Define CUDA language
AC_LANG_DEFINE([CUDA], [cuda], [NVCC], [NVCC], [C++], [
    ac_ext=cu
    ac_compile="$NVCC $BASE_NVCCFLAGS $NVCCFLAGS -c -o conftest.o conftest.$ac_ext"
    ac_link="$NVCC $BASE_NVCCFLAGS $NVCCFLAGS -o conftest conftest.o"
   ],
   [rm -f conftest.o conftest.$ac_ext conftest])

# Define CUDA language compiler
AC_DEFUN([AC_LANG_COMPILER(CUDA)], [
    AC_ARG_VAR([NVCC], [nvcc compiler path])
    AC_ARG_VAR([NVCCFLAGS], [nvcc compiler flags])
    BASE_NVCCFLAGS="$BASE_NVCCFLAGS -g $with_nvcc_gencode"
    AS_IF([test ! -z "$with_cuda" -a -d "$with_cuda/bin"],
          [CUDA_BIN_PATH="$with_cuda/bin"],
          [CUDA_BIN_PATH=""])
    AC_PATH_PROG([NVCC], [nvcc], [], [$CUDA_BIN_PATH:$PATH])
    AC_SUBST([NVCC], [$NVCC])
])

# Check for nvcc compiler support
AC_DEFUN([UCX_CUDA_CHECK_NVCC], [
    AS_IF([test "x$NVCC" != "x"], [
        CUDA_MAJOR_VERSION=`$NVCC --version | grep release | sed 's/.*release //' | sed 's/\,.*//' |  cut -d "." -f 1`
        CUDA_MINOR_VERSION=`$NVCC --version | grep release | sed 's/.*release //' | sed 's/\,.*//' |  cut -d "." -f 2`
        AC_MSG_RESULT([Detected CUDA version: $CUDA_MAJOR_VERSION.$CUDA_MINOR_VERSION])
        AS_IF([test $CUDA_MAJOR_VERSION -lt $NVCC_CUDA_MIN_REQUIRED_MAJOR -o \( $CUDA_MAJOR_VERSION -eq $NVCC_CUDA_MIN_REQUIRED_MAJOR -a $CUDA_MINOR_VERSION -lt $NVCC_CUDA_MIN_REQUIRED_MINOR \)],
            [AC_MSG_WARN([Minimum required CUDA version for device code: $NVCC_CUDA_MIN_REQUIRED_MAJOR.$NVCC_CUDA_MIN_REQUIRED_MINOR])
             NVCC=""
            ])

        AS_IF([test "x$NVCC" != "x"], [
                AC_ARG_WITH([nvcc-gencode],
                            [AS_HELP_STRING([--with-nvcc-gencode=(OPTS)], [Build for specific GPU architectures])],
                            [],
                            [with_nvcc_gencode=default])

                AS_IF([test "x$with_nvcc_gencode" = "xdefault"],
                        [AS_CASE([$CUDA_MAJOR_VERSION],
                                 [13],
                                     [# offline compilation support for architectures before '<compute/sm/lto>_75' is discontinued
                                      NVCC_ARCH="${ARCH10_CODE} ${ARCH110_CODE} ${ARCH111_CODE} ${ARCH120_CODE} ${ARCH124_CODE} ${ARCH128_CODE} ${ARCH130_CODE} ${ARCH130_PTX}"],
                                 [12],
                                     [AS_CASE([$CUDA_MINOR_VERSION],
                                              [0|1|2|3],
                                                  [NVCC_ARCH="${ARCH9_CODE} ${ARCH10_CODE} ${ARCH110_CODE} ${ARCH111_CODE} ${ARCH120_CODE} ${ARCH120_PTX}"],
                                              [4|5|6|7],
                                                  [NVCC_ARCH="${ARCH9_CODE} ${ARCH10_CODE} ${ARCH110_CODE} ${ARCH111_CODE} ${ARCH120_CODE} ${ARCH124_CODE} ${ARCH124_PTX}"],
                                              [*],
                                                  [NVCC_ARCH="${ARCH9_CODE} ${ARCH10_CODE} ${ARCH110_CODE} ${ARCH111_CODE} ${ARCH120_CODE} ${ARCH124_CODE} ${ARCH128_CODE} ${ARCH128_PTX}"])],

                                 [11],
                                     [AS_CASE([$CUDA_MINOR_VERSION],
                                              [0],
                                                  [NVCC_ARCH="${ARCH9_CODE} ${ARCH10_CODE} ${ARCH110_CODE} ${ARCH110_PTX}"],
                                              [*],
                                                  [NVCC_ARCH="${ARCH9_CODE} ${ARCH10_CODE} ${ARCH110_CODE} ${ARCH111_CODE} ${ARCH111_PTX}"])],
                                 [*],
                                     [NVCC_ARCH="${ARCH9_CODE} ${ARCH9_PTX}"])],
                        [NVCC_ARCH="$with_nvcc_gencode"])
                BASE_NVCCFLAGS="$BASE_NVCCFLAGS $NVCC_ARCH"
                AC_MSG_CHECKING([$NVCC needs explicit c++11 option])
                AC_LANG_PUSH([CUDA])
                AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
                    #if __cplusplus < 201103L
                    #error missing C++11
                    #endif
                ]])],
                [AC_MSG_RESULT([no])],
                [AC_MSG_RESULT([yes])
                BASE_NVCCFLAGS="$BASE_NVCCFLAGS -std=c++11"])
                AC_LANG_POP

                AC_MSG_CHECKING([$NVCC can compile])
                AC_LANG_PUSH([CUDA])
                AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
                    #include <cuda_runtime.h>
                    __global__ void my_kernel(void) {}
                    int main(void) { my_kernel<<<1, 1>>>(); return 0; }
                ]])],
                [AC_MSG_RESULT([yes])],
                [AC_MSG_RESULT([no])
                NVCC=""])
                AC_LANG_POP
            ])
        ])

    AM_CONDITIONAL([HAVE_NVCC], [test "x$NVCC" != x])
])

# Check for CUDA support
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
         CUDA_LIB_DIRS=""

         AS_IF([test ! -z "$with_cuda" -a "x$with_cuda" != "xyes" -a "x$with_cuda" != "xguess"],
               [ucx_check_cuda_dir="$with_cuda"
                AS_IF([test -d "$with_cuda/lib64"], [libsuff="64"], [libsuff=""])
                ucx_check_cuda_libdir="$with_cuda/lib$libsuff"
                CUDA_CPPFLAGS="-I$with_cuda/include"
                CUDA_LDFLAGS="-L$ucx_check_cuda_libdir -L$ucx_check_cuda_libdir/stubs"
                CUDA_LIB_DIRS="$ucx_check_cuda_libdir $with_cuda/compat"])

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

        UCX_CUDA_CHECK_NVCC
   ]) # "x$cuda_checked" != "xyes"

]) # UCX_CHECK_CUDA
