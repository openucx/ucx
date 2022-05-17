#
# Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

AC_LANG_PUSH([C++])

CHECK_COMPILER_FLAG([-fno-tree-vectorize], [-fno-tree-vectorize],
                    [AC_LANG_SOURCE([[int main(int argc, char** argv){return 0;}]])],
                    [GTEST_CXXFLAGS="$GTEST_CXXFLAGS -fno-tree-vectorize"],
                    [])

# error #186: pointless comparison of unsigned integer with zero
CHECK_COMPILER_FLAG([--diag_suppress 186], [--diag_suppress 186],
                    [AC_LANG_SOURCE([[int main(int argc, char** argv){return 0;}]])],
                    [GTEST_CXXFLAGS="$GTEST_CXXFLAGS --diag_suppress 186"],
                    [])
                    
# error #236: controlling expression is constant
CHECK_COMPILER_FLAG([--diag_suppress 236], [--diag_suppress 236],
                    [AC_LANG_SOURCE([[int main(int argc, char** argv){return 0;}]])],
                    [GTEST_CXXFLAGS="$GTEST_CXXFLAGS --diag_suppress 236"],
                    [])

AC_LANG_POP([C++])

AC_SUBST([GTEST_CXXFLAGS], [$GTEST_CXXFLAGS])

test_modules=""
m4_include([test/gtest/common/googletest/configure.m4])
m4_include([test/gtest/ucm/test_dlopen/configure.m4])
m4_include([test/gtest/ucm/test_dlopen/rpath-subdir/configure.m4])
m4_include([test/gtest/ucs/test_module/configure.m4])
AC_DEFINE_UNQUOTED([test_MODULES], ["${test_modules}"], [Test loadable modules])
AC_CONFIG_FILES([test/gtest/Makefile])
