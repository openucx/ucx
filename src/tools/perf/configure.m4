#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

ucx_perftest_modules=""
m4_include([src/tools/perf/lib/configure.m4])
m4_include([src/tools/perf/cuda/configure.m4])
m4_include([src/tools/perf/rocm/configure.m4])
AC_DEFINE_UNQUOTED([ucx_perftest_MODULES], ["${ucx_perftest_modules}"],
                   [Perftest loadable modules])

# TODO build RTE support (MPI/librte) as loadable modules
AS_IF([test -n "$MPICC"],
      [AC_SUBST([UCX_PERFTEST_CC], [$MPICC])],
      [AC_SUBST([UCX_PERFTEST_CC], [$CC])])

AC_CONFIG_FILES([src/tools/perf/Makefile])
