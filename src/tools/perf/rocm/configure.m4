#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
# Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

UCX_CHECK_ROCM

# HIP version check for tools/perf/rocm. In HIP version 1.5, hip_runtime.h can
# trigger error due to missing braces around initializer in hip_vector_type.h

AC_CHECK_PROG([HIPCONFIG_CHECK], [hipconfig], [yes], [], [$with_rocm/hip/bin])

AC_MSG_CHECKING([HIP version for ROCm perftest])
hip_happy=no
if test x"${HIPCONFIG_CHECK}" == x"yes" ; then
    HIPCONFIG=${with_rocm}/hip/bin/hipconfig
    HIP_VER_MAJOR=$($HIPCONFIG -v | cut -d '.' -f1)
    HIP_VER_MINOR=$($HIPCONFIG -v | cut -d '.' -f2)
    AS_VERSION_COMPARE([$HIP_VER_MAJOR.$HIP_VER_MINOR], [2.0],
        [AC_MSG_RESULT(
            [HIP v${HIP_VER_MAJOR}.${HIP_VER_MINOR} is old, skipping ROCm perftest])],
        [hip_happy=yes],
        [hip_happy=yes]
        )
else
    AC_MSG_RESULT([no hipconfig detected, skipping ROCm perftest])
fi

AS_IF([test "x$rocm_happy" = "xyes" && test "x$hip_happy" = "xyes"],
    [AC_MSG_RESULT([yes])
     ucx_perftest_modules="${ucx_perftest_modules}:rocm"]
)

AC_CONFIG_FILES([src/tools/perf/rocm/Makefile])
