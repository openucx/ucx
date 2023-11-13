#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2018. ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

AC_LANG_PUSH([C++])

UCX_CHECK_CUDA
UCX_CHECK_ROCM
AS_IF([test "x$cuda_happy" = "xyes"], [ucx_perftest_modules="${ucx_perftest_modules}:cuda"])
AS_IF([test "x$rocm_happy" = "xyes"], [ucx_perftest_modules="${ucx_perftest_modules}:rocm"])

CHECK_COMPILER_FLAG([-fno-exceptions], [-fno-exceptions],
                    [AC_LANG_SOURCE([[int main(int argc, char** argv){return 0;}]])],
                    [PERF_LIB_CXXFLAGS="$PERF_LIB_CXXFLAGS -fno-exceptions"],
                    [])

CHECK_COMPILER_FLAG([-fno-rtti], [-fno-rtti],
                    [AC_LANG_SOURCE([[int main(int argc, char** argv){return 0;}]])],
                    [PERF_LIB_CXXFLAGS="$PERF_LIB_CXXFLAGS -fno-rtti"],
                    [])

CHECK_COMPILER_FLAG([--no_exceptions], [--no_exceptions],
                    [AC_LANG_SOURCE([[int main(int argc, char** argv){return 0;}]])],
                    [PERF_LIB_CXXFLAGS="$PERF_LIB_CXXFLAGS --no_exceptions"],
                    [])

AC_LANG_POP([C++])

AC_SUBST([PERF_LIB_CXXFLAGS], [$PERF_LIB_CXXFLAGS])

AC_CONFIG_FILES([src/tools/perf/lib/Makefile])
