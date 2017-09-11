#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
# Copyright (c) UT-Battelle, LLC. 2017. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

#
# Initialize CFLAGS
#
BASE_CFLAGS="-g -Wall -Werror"

#
# Debug mode
#
AC_ARG_ENABLE(debug,
        AC_HELP_STRING([--enable-debug], [Enable debug mode build]),
        [],
        [enable_debug=no])
AS_IF([test "x$enable_debug" == xyes],
        [BASE_CFLAGS="-D_DEBUG $BASE_CFLAGS"],
        [])

#
# Optimization level
#
AC_ARG_ENABLE(compiler-opt,
        AC_HELP_STRING([--enable-compiler-opt], [Set optimization level [0-3]]),
        [],
        [enable_compiler_opt="none"])
AS_IF([test "$enable_compiler_opt" == "yes"], [BASE_CFLAGS="-O3 $BASE_CFLAGS"],
      [test "$enable_compiler_opt" == "none"],
          [AS_IF([test "x$enable_debug" == xyes],
                 [BASE_CFLAGS="-O0 $BASE_CFLAGS"],
                 [BASE_CFLAGS="-O3 $BASE_CFLAGS"])],
      [test "$enable_compiler_opt" == "no"], [],
      [BASE_CFLAGS="-O$enable_compiler_opt $BASE_CFLAGS"])


#
# CHECK_CROSS_COMP (program, true-action, false-action)
#
# The macro checks if it can run the program; it executes
# true action if the program can be executed, otherwise
# false action is executed.
# For cross-platform compilation we only check
# if we can compile and link the program.
AC_DEFUN([CHECK_CROSS_COMP], [
         AC_RUN_IFELSE([$1], [$2], [$3],
                       [AC_LINK_IFELSE([$1], [$2], [$3])])
])

#
# Check for one specific attribute by compiling with C
# Usage: CHECK_SPECIFIC_ATTRIBUTE([name], [doc], [program])
#
AC_DEFUN([CHECK_SPECIFIC_ATTRIBUTE], [
    AC_CACHE_VAL(ucx_cv_attribute_[$1], [
        SAVE_CFLAGS="$CFLAGS"
        CFLAGS="$BASE_CFLAGS $CFLAGS"
        #
        # Try to compile using the C compiler
        #
        AC_TRY_COMPILE([$3],[],
                       [ucx_cv_attribute_[$1]=1],
                       [ucx_cv_attribute_[$1]=0])
	CFLAGS="$SAVE_CFLAGS"
    ])
	AC_MSG_CHECKING([for __attribute__([$1])])
	AC_MSG_RESULT([$ucx_cv_attribute_[$1]])
	AC_DEFINE_UNQUOTED([HAVE_ATTRIBUTE_[$2]], [$ucx_cv_attribute_[$1]], [Check attribute [$1]])
])

#
# Check if compiler supports a given feaure
# Usage: COMPILER_OPTION([name], [doc], [flag], [default: yes|no], [program])
#
AC_DEFUN([COMPILER_OPTION],
[
    AC_ARG_WITH([$1],
                [AC_HELP_STRING([--with-$1], [Use $2 compiler option.])],
                [],
                [with_$1=$4])
   
    AS_IF([test "x$with_$1" != "xno"],
          [SAVE_CFLAGS="$CFLAGS"
           CFLAGS="$BASE_CFLAGS $CFLAGS $3"
           AC_MSG_CHECKING([$2])
           CHECK_CROSS_COMP([AC_LANG_SOURCE([$5])],
                            [AC_MSG_RESULT([yes])
                             OPT_CFLAGS="$OPT_CFLAGS|$1"],
                            [AC_MSG_RESULT([no])])
           CFLAGS="$SAVE_CFLAGS"])
])


#
# CHECK_DEPRECATED_DECL_FLAG (flag, variable)
#
# The macro checks if the given compiler flag enables usig deprecated declarations.
# If yes, it appends the flags to "variable".
#
AC_DEFUN([CHECK_DEPRECATED_DECL_FLAG],
[
         AC_MSG_CHECKING([whether $1 overrides deprecated declarations])
         SAVE_CFLAGS="$CFLAGS"
         CFLAGS="$BASE_CFLAGS $CFLAGS $1"
         AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
                                  int __attribute__ ((__deprecated__)) f() { return 0; }
                                  int main() { return f(); }
                            ]])],
                           [AC_MSG_RESULT([yes])
                            $2="${$2} $1"],
                           [AC_MSG_RESULT([no])])
         CFLAGS="$SAVE_CFLAGS"
])

CHECK_DEPRECATED_DECL_FLAG([-diag-disable 1478], CFLAGS_NO_DEPRECATED) # icc
CHECK_DEPRECATED_DECL_FLAG([-Wno-deprecated-declarations], CFLAGS_NO_DEPRECATED) # gcc
AC_SUBST([CFLAGS_NO_DEPRECATED], [$CFLAGS_NO_DEPRECATED])


#
# Disable format-string warning on ICC
#
SAVE_CFLAGS="$CFLAGS"
CFLAGS="$BASE_CFLAGS $CFLAGS -diag-disable 269"
AC_MSG_CHECKING([-diag-disable 269])
AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
                     #include <stdlib.h>
                     #include <stdio.h>
                     int main() {
                         char *p = NULL;
                         scanf("%m[^.]", &p);
                         free(p);
                         return 0;
                     }
                 ]])],
               [AC_MSG_RESULT([yes])],
               [AC_MSG_RESULT([no])
                CFLAGS="$SAVE_CFLAGS"])


#
#  Enable/disable turning on machine-specific optimizations
#
AC_ARG_ENABLE(optimizations,
        AC_HELP_STRING([--enable-optimizations], [Enable machine-specific optimizations, default: NO]),
        [],
        [enable_optimizations=no])


#
# SSE/AVX
#
COMPILER_OPTION([avx], [AVX], [-mavx], [$enable_optimizations],
                [#include <immintrin.h>
                 int main() { return _mm256_testz_si256(_mm256_set1_epi32(1), _mm256_set1_epi32(3)); }])
AS_IF([test "x$with_avx" != xyes],
      [COMPILER_OPTION([sse41], [SSE 4.1], [-msse4.1], [$enable_optimizations],
                       [#include <smmintrin.h>
                       int main() { return _mm_testz_si128(_mm_set1_epi32(1), _mm_set1_epi32(3)); }])
       COMPILER_OPTION([sse42], [SSE 4.2], [-msse4.2], [$enable_optimizations],
                       [#include <popcntintrin.h>
                        int main() { return _mm_popcnt_u32(0x101) - 2; }])
      ])


#
# Check for compiler attribute which disables optimizations per-function.
#
CHECK_SPECIFIC_ATTRIBUTE([optimize], [NOOPTIMIZE],
                         [int foo (int arg) __attribute__ ((optimize("O0")));])


#
# Set C++ optimization/debug flags to be the same as for C
#
BASE_CPPFLAGS="-DCPU_FLAGS=\"$OPT_CFLAGS\""
BASE_CXXFLAGS="$BASE_CFLAGS"

AC_SUBST([BASE_CPPFLAGS], [$BASE_CPPFLAGS])
AC_SUBST([BASE_CFLAGS], [$BASE_CFLAGS]) 
AC_SUBST([BASE_CXXFLAGS], [$BASE_CXXFLAGS])

