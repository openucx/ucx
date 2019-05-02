#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
# Copyright (c) UT-Battelle, LLC. 2017. ALL RIGHTS RESERVED.
# Copyright (C) ARM Ltd. 2016-2018.  ALL RIGHTS RESERVED.
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
AS_IF([test "x$enable_debug" = xyes],
        [BASE_CFLAGS="-D_DEBUG $BASE_CFLAGS"],
        [])

#
# Optimization level
#
AC_ARG_ENABLE(compiler-opt,
        AC_HELP_STRING([--enable-compiler-opt], [Set optimization level [0-3]]),
        [],
        [enable_compiler_opt="none"])
AS_IF([test "x$enable_compiler_opt" = "xyes"], [BASE_CFLAGS="-O3 $BASE_CFLAGS"],
      [test "x$enable_compiler_opt" = "xnone"],
          [AS_IF([test "x$enable_debug" = xyes],
                 [BASE_CFLAGS="-O0 $BASE_CFLAGS"],
                 [BASE_CFLAGS="-O3 $BASE_CFLAGS"])],
      [test "x$enable_compiler_opt" = "xno"], [],
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
           AC_MSG_CHECKING([$3])
           CHECK_CROSS_COMP([AC_LANG_SOURCE([$5])],
                            [AC_MSG_RESULT([yes])
			     # TODO: Add CPU UARCH detector and validator in UCX init.
			     # As for now we will avoid passing this information to
			     # library.
			     AS_IF([test "x$1" != "xmcpu" -a "x$1" != "xmarch"],
                             [OPT_CFLAGS="$OPT_CFLAGS|$1"],[])],
                            [AC_MSG_RESULT([no])])
           CFLAGS="$SAVE_CFLAGS"])
])


#
# Check platform uarch and apply micro-architecture specific optimizations
#
AC_DEFUN([DETECT_UARCH],
[
    cpuimpl=`grep 'CPU implementer' /proc/cpuinfo 2> /dev/null | cut -d: -f2 | tr -d " " | head -n 1`
    cpuarch=`grep 'CPU architecture' /proc/cpuinfo 2> /dev/null | cut -d: -f2 | tr -d " " | head -n 1`
    cpuvar=`grep 'CPU variant' /proc/cpuinfo 2> /dev/null | cut -d: -f2 | tr -d " " | head -n 1`
    cpupart=`grep 'CPU part' /proc/cpuinfo 2> /dev/null | cut -d: -f2 | tr -d " " | head -n 1`
   
    ax_cpu=""
    ax_arch=""
    
    AC_MSG_NOTICE(Detected CPU implementation: ${cpuimpl})
    AC_MSG_NOTICE(Detected CPU architecture: ${cpuarch})
    AC_MSG_NOTICE(Detected CPU variant: ${cpuvar})
    AC_MSG_NOTICE(Detected CPU part: ${cpupart})
   
    case $cpuimpl in
      0x42) case $cpupart in
        0x516 | 0x0516)
          AC_DEFINE([HAVE_AARCH64_THUNDERX2], 1, [Cavium ThunderX2])
          ax_cpu="thunderx2t99"
          ax_arch="armv8.1-a+lse" ;;
        0xaf | 0x0af)
          AC_DEFINE([HAVE_AARCH64_THUNDERX2], 1, [Cavium ThunderX2])
          ax_cpu="thunderx2t99"
          ax_arch="armv8.1-a+lse" ;;
        esac
        ;;
      0x43) case $cpupart in
        0x516 | 0x0516)
          AC_DEFINE([HAVE_AARCH64_THUNDERX2], 1, [Cavium ThunderX2])
          ax_cpu="thunderx2t99"
          ax_arch="armv8.1-a+lse" ;;
        0xaf | 0x0af)
          AC_DEFINE([HAVE_AARCH64_THUNDERX2], 1, [Cavium ThunderX2])
          ax_cpu="thunderx2t99"
          ax_arch="armv8.1-a+lse" ;;
        0xa1 | 0x0a1)
          AC_DEFINE([HAVE_AARCH64_THUNDERX1], 1, [Cavium ThunderX1])
          ax_cpu="thunderxt88" ;;
        esac
        ;;
      *) ax_cpu="native"
         ;;
    esac 
])


#
# CHECK_COMPILER_FLAG
# Usage: CHECK_COMPILER_FLAG([name], [flag], [program], [if-true], [if-false])
#
# The macro checks if program may be compiled using specified flag
#
AC_DEFUN([CHECK_COMPILER_FLAG],
[
#
# Force ICC treat command line warnings as errors.
# This evaluation should be called prior to all other compiler flags evals
#
         AS_IF([test "x$icc_cmd_diag_to_error" = "x"],
               [icc_cmd_diag_to_error=1
                AC_MSG_CHECKING([compiler flag -diag-error 10006])
                SAVE_CFLAGS="$CFLAGS"
                CFLAGS="$BASE_CFLAGS $CFLAGS -diag-error 10006"
                AC_COMPILE_IFELSE([AC_LANG_SOURCE([[int main(){return 0;}]])],
                                  [BASE_CFLAGS="$BASE_CFLAGS -diag-error 10006"
                                   AC_MSG_RESULT([yes])],
                                  [AC_MSG_RESULT([no])])
                CFLAGS="$SAVE_CFLAGS"
               ],
               [])
         AC_MSG_CHECKING([compiler flag $1])
         SAVE_CFLAGS="$CFLAGS"
         CFLAGS="$BASE_CFLAGS $CFLAGS $2"
         AC_COMPILE_IFELSE([$3],
                           [AC_MSG_RESULT([yes])
                            $4],
                           [AC_MSG_RESULT([no])
                            $5])
         CFLAGS="$SAVE_CFLAGS"
])

#
# ADD_COMPILER_FLAG_IF_SUPPORTED
# Usage: ADD_COMPILER_FLAG_IF_SUPPORTED([name], [flag], [program], [if-true], [if-false])
#
# The macro checks if program may be compiled using specified flag and adds
# this flag if it is supported
#
AC_DEFUN([ADD_COMPILER_FLAG_IF_SUPPORTED],
[
         CHECK_COMPILER_FLAG([$1], [$2], [$3],
                             [BASE_CFLAGS="$BASE_CFLAGS $2" $4],
                             [$5])
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
ADD_COMPILER_FLAG_IF_SUPPORTED([-diag-disable 269],
                               [-diag-disable 269],
                               [AC_LANG_SOURCE([[#include <stdlib.h>
                                                 #include <stdio.h>
                                                 int main() {
                                                     char *p = NULL;
                                                     scanf("%m[^.]", &p);
                                                     free(p);
                                                     return 0;
                                                 }]])],
                               [],
                               [])


#
# Set default datatype alignment to 16 bytes.
# Some compilers (LLVM based, clang) expects allocation of datatypes by 32 bytes
# to optimize operations memset/memcpy/etc using vectorized processor instructions
# which requires aligment of memory buffer by 32 or higer bytes. Default malloc method
# guarantee alignment for 16 bytes only. Force using compiler 16-bytes alignment
# by default if option is supported.
#
UCX_ALLOC_ALIGN=16
ADD_COMPILER_FLAG_IF_SUPPORTED([-fmax-type-align=$UCX_ALLOC_ALIGN],
                               [-fmax-type-align=$UCX_ALLOC_ALIGN],
                               [AC_LANG_SOURCE([[int main(){return 0;}]])],
                               [AC_DEFINE_UNQUOTED([UCX_ALLOC_ALIGN], $UCX_ALLOC_ALIGN, [Set aligment assumption for compiler])],
                               [])


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


DETECT_UARCH()

#
# CPU tuning
#
AS_IF([test "x$ax_cpu" != "x"],
      [COMPILER_OPTION([mcpu], [CPU Model], [-mcpu=$ax_cpu], [$enable_optimizations],
		 [int main() { return 0;}])
      ])

# 
# Architecture tuning
# 
AS_IF([test "x$ax_arch" != "x"],
      [COMPILER_OPTION([march], [architecture tuning], [-march=$ax_arch], [$enable_optimizations],
		 [int main() { return 0;}])
      ])


#
# Check for compiler attribute which disables optimizations per-function.
#
CHECK_SPECIFIC_ATTRIBUTE([optimize], [NOOPTIMIZE],
                         [int foo (int arg) __attribute__ ((optimize("O0")));])


#
# Check for C++11 support
#
AC_MSG_CHECKING([c++11 support])
AC_LANG_PUSH([C++])
SAVE_CXXFLAGS="$CXXFLAGS"
CXX11FLAGS="-std=c++11"
CXXFLAGS="$CXXFLAGS $CXX11FLAGS"
AC_COMPILE_IFELSE([AC_LANG_SOURCE([[#include <iostream>
					#include <string>
					int main() {
						std::string my_str;
						int a = 5;
						my_str = std::to_string(a);
						std::cout << "to_string: " << my_str << '\n';
						return 0;
					} ]])],
                  [AC_MSG_RESULT([yes])
                   AC_SUBST([CXX11FLAGS])
                   cxx11_happy=yes],
                  [AC_MSG_RESULT([no])
                   cxx11_happy=no])
CXXFLAGS="$SAVE_CXXFLAGS"
AC_LANG_POP
AM_CONDITIONAL([HAVE_CXX11], [test "x$cxx11_happy" != xno])


#
# Set C++ optimization/debug flags to be the same as for C
#
BASE_CXXFLAGS="$BASE_CFLAGS"
AC_SUBST([BASE_CFLAGS], [$BASE_CFLAGS]) 
AC_SUBST([BASE_CXXFLAGS], [$BASE_CXXFLAGS])

#
# Set common preprocessor flags
#
BASE_CPPFLAGS="-DCPU_FLAGS=\"$OPT_CFLAGS\""
BASE_CPPFLAGS="$BASE_CPPFLAGS -I\${abs_top_srcdir}/src"
BASE_CPPFLAGS="$BASE_CPPFLAGS -I\${abs_top_builddir}"
BASE_CPPFLAGS="$BASE_CPPFLAGS -I\${abs_top_builddir}/src"
AC_MSG_NOTICE([Common preprocessor flags: ${BASE_CPPFLAGS}])
AC_SUBST([BASE_CPPFLAGS], [$BASE_CPPFLAGS])
