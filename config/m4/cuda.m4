#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2017. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

NVCC_CUDA_MIN_REQUIRED=12.2

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
    BASE_NVCCFLAGS="$BASE_NVCCFLAGS -g"
    AS_IF([test ! -z "$with_cuda" -a -d "$with_cuda/bin"],
          [CUDA_BIN_PATH="$with_cuda/bin"],
          [CUDA_BIN_PATH=""])
    AC_PATH_PROG([NVCC], [nvcc], [], [$CUDA_BIN_PATH:$PATH])
    AC_SUBST([NVCC], [$NVCC])
])

AC_DEFUN([UCX_CUDA_CHECK_NVCC_GENCODE], [
    pattern=$1
    min_version=$2
    AC_MSG_CHECKING([NVCC code generation options])
    NVCC_GENCODE=""
    for codegen in $($NVCC --list-gpu-code)
    do
        case "$codegen" in
        $pattern)
            version=${codegen#sm_}
            if test $version -ge ${min_version}
            then
                 NVCC_GENCODE="$NVCC_GENCODE -gencode=arch=compute_${version},code=${codegen}"
            fi
            ;;
        esac
    done
    NVCC_GENCODE=${NVCC_GENCODE# }
    AS_IF([test "x$NVCC_GENCODE" = "x"],
          [AC_MSG_ERROR([No NVCC code generation options found for pattern: $pattern and min_version: $min_version])],
          [AC_MSG_RESULT([$NVCC_GENCODE])
           BASE_NVCCFLAGS="$BASE_NVCCFLAGS $NVCC_GENCODE"])
])

# Check for nvcc compiler support
AC_DEFUN([UCX_CUDA_CHECK_NVCC], [
    AS_IF([test "x$NVCC" != "x"], [
        CUDA_VERSION=$($NVCC --version | grep release | sed 's/.*release //' | sed 's/\,.*//')
        CUDA_MAJOR_VERSION=$(echo $CUDA_VERSION | cut -d "." -f 1)
        CUDA_MINOR_VERSION=$(echo $CUDA_VERSION | cut -d "." -f 2)
        AC_MSG_RESULT([Detected CUDA version: $CUDA_VERSION])
        AS_VERSION_COMPARE([$CUDA_VERSION], [$NVCC_CUDA_MIN_REQUIRED],
              [AC_MSG_WARN([Minimum required CUDA version for device code: $NVCC_CUDA_MIN_REQUIRED])
               NVCC=""])

        NVCC_CXX_DIALECT=c++17
        cxx_dialect_ver=201703L
        AS_VERSION_COMPARE([$CUDA_VERSION], [13.0],
              [NVCC_CXX_DIALECT=c++11
               cxx_dialect_ver=201103L
               NVCCFLAGS="$NVCCFLAGS -DCCCL_IGNORE_DEPRECATED_CPP_DIALECT"])

        AS_VERSION_COMPARE([$CUDA_VERSION], [12.9], [],
              [NVCCFLAGS="$NVCCFLAGS -D_LIBCUDACXX_ATOMIC_UNSAFE_AUTOMATIC_STORAGE"],
              [NVCCFLAGS="$NVCCFLAGS -D_LIBCUDACXX_ATOMIC_UNSAFE_AUTOMATIC_STORAGE"])

        AS_IF([test "x$NVCC" != "x"], [
                AC_ARG_WITH([nvcc-arch],
                            [AS_HELP_STRING([--with-nvcc-arch=(ARCH)],
                             [Build for specific GPU architecture, for example: 'sm_80'.
                              Use 'all' to build for all supported architectures.
                              Use 'all-major' to build for all supported major architectures.])],
                            [],
                            [with_nvcc_gencode=all-major])

                nvcc_min_arch=80
                AS_CASE([$with_nvcc_gencode],
                        [all],        [UCX_CUDA_CHECK_NVCC_GENCODE([sm_*], [$nvcc_min_arch])],
                        [all-major],  [UCX_CUDA_CHECK_NVCC_GENCODE([sm_*0], [$nvcc_min_arch])],
                        [yes|native], [BASE_NVCCFLAGS="$BASE_NVCCFLAGS -arch=native"],
                        [*],          [BASE_NVCCFLAGS="$BASE_NVCCFLAGS -arch=$with_nvcc_arch"])

                AC_MSG_CHECKING([$NVCC needs explicit $NVCC_CXX_DIALECT option])
                AC_LANG_PUSH([CUDA])
                AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
                    #if __cplusplus < $cxx_dialect_ver
                    #error missing $NVCC_CXX_DIALECT
                    #endif
                ]])],
                [AC_MSG_RESULT([no])],
                [AC_MSG_RESULT([yes])
                BASE_NVCCFLAGS="$BASE_NVCCFLAGS -std=$NVCC_CXX_DIALECT"])
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

                AC_MSG_CHECKING([checking cuda/atomic support])
                AC_LANG_PUSH([CUDA])
                AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
                      #include <cuda/atomic>
                      int v;
                      cuda::atomic_ref<int> ref{v};
                   ]])],
                   [AC_MSG_RESULT([yes])],
                   [AC_MSG_RESULT([no])
                    NVCC=""])
                AC_LANG_POP

                # Check curand_kernel.h (optional, required for random channel mode)
                AC_MSG_CHECKING([for curand_kernel.h])
                AC_LANG_PUSH([CUDA])
                AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
                    #include <curand_kernel.h>
                    __global__ void test(curandState *s) { curand_init(0, 0, 0, s); }
                ]])],
                [AC_MSG_RESULT([yes])
                 AC_DEFINE([HAVE_CURAND], [1], [cuRAND device API is available])],
                [AC_MSG_RESULT([no])
                 AC_MSG_NOTICE([curand_kernel.h not found. Install libcurand-devel to enable random channel mode in perftest.])])
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

         AC_CHECK_DECLS([NVML_FI_DEV_C2C_LINK_COUNT], [], [],
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
