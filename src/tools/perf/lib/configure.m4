#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

AC_LANG_PUSH([C++])

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
