#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

#
# Initialize CFLAGS
#
CFLAGS="-g -Wall -Werror $UCX_CFLAGS"


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
           CFLAGS="$CFLAGS $3"
           AC_MSG_CHECKING([$2])
           CHECK_CROSS_COMP([AC_LANG_SOURCE([$5])],
                            [AC_MSG_RESULT([yes])],
                            [AC_MSG_RESULT([no])
                             CFLAGS="$SAVE_CFLAGS"])])
])


#
# Debug mode
#
AC_ARG_ENABLE(debug,
        AC_HELP_STRING([--enable-debug], [Enable debug mode build]),
        [],
        [enable_debug=no])
AS_IF([test "x$enable_debug" == xyes],
        [CFLAGS="-O0 -D_DEBUG $CFLAGS"],
        [CFLAGS="-O3 $CFLAGS"])


#
# SSE/AVX
#
COMPILER_OPTION([sse41], [SSE 4.1], [-msse4.1], [yes],
                [#include <smmintrin.h>
                 int main() { return _mm_testz_si128(_mm_set1_epi32(1), _mm_set1_epi32(3)); }])
COMPILER_OPTION([avx], [AVX], [-mavx], [yes],
                [#include <immintrin.h>
                 int main() { return _mm256_testz_si256(_mm256_set1_epi32(1), _mm256_set1_epi32(3)); }])

#
# Set C++ optimization/debug flags to be the same as for C
#
CXXFLAGS="$CFLAGS"
