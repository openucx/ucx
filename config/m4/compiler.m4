#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2014. ALL RIGHTS RESERVED.
# Copyright (c) UT-Battelle, LLC. 2017. ALL RIGHTS RESERVED.
# Copyright (C) ARM Ltd. 2016-2020.  ALL RIGHTS RESERVED.
# Copyright (C) NextSilicon Ltd. 2021.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#


#
# Initialize CFLAGS
#
BASE_CFLAGS="-g -Wall -Werror"


#
# Check that C++ is functional.
#
# AC_PROG_CXX never fails but falls back on g++ as a default CXX compiler that
# always present. If g++ isn't installed, the macro doesn't detect this and
# compilation fails later on. CHECK_CXX_COMP compiles simple C++ code to
# verify that compiler is present and functional.
#
AC_DEFUN([CHECK_CXX_COMP],
         [AC_MSG_CHECKING(if $CXX works)
          AC_LANG_PUSH([C++])
          AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
                            #ifndef __cplusplus
                            #error "No C++ support, AC_PROG_CXX failed"
                            #endif
                            ]])],
                            [AC_MSG_RESULT([yes])],
                            [AC_MSG_ERROR([Cannot continue. Please install C++ compiler.])])
          AC_LANG_POP([C++])
         ])


#
# Debug mode
#
AC_ARG_ENABLE(debug,
        AS_HELP_STRING([--enable-debug], [Enable debug mode build]),
        [],
        [enable_debug=no])
AS_IF([test "x$enable_debug" = xyes],
        [BASE_CFLAGS="-D_DEBUG $BASE_CFLAGS"
         BASE_CXXFLAGS="-D_DEBUG" $BASE_CXXFLAGS],
        [])


#
# Enable GCOV build
#
AC_ARG_ENABLE([gcov],
        AS_HELP_STRING([--enable-gcov], [Enable code coverage instrumentation]),
        [],
        [enable_gcov=no])
AM_CONDITIONAL([HAVE_GCOV],[test "x$enable_gcov" = xyes])


#
# Optimization level
#
AC_ARG_ENABLE(compiler-opt,
        AS_HELP_STRING([--enable-compiler-opt], [Set optimization level [0-3]]),
        [],
        [enable_compiler_opt="none"])
AS_IF([test "x$enable_compiler_opt" = "xyes"], [BASE_CFLAGS="-O3 $BASE_CFLAGS"],
      [test "x$enable_compiler_opt" = "xnone"],
          [AS_IF([test "x$enable_debug" = xyes -o "x$enable_gcov" = xyes],
                 [BASE_CFLAGS="-O0 $BASE_CFLAGS"
                  BASE_CXXFLAGS="-O0 $BASE_CXXFLAGS"],
                 [BASE_CFLAGS="-O3 $BASE_CFLAGS"
                  BASE_CXXFLAGS="-O0 $BASE_CXXFLAGS"])],
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
        AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[$3]],[[]])],
                       [ucx_cv_attribute_[$1]=1],
                       [ucx_cv_attribute_[$1]=0])
	CFLAGS="$SAVE_CFLAGS"
    ])
	AC_MSG_CHECKING([for __attribute__([$1])])
	AC_MSG_RESULT([$ucx_cv_attribute_[$1]])
	AC_DEFINE_UNQUOTED([HAVE_ATTRIBUTE_[$2]], [$ucx_cv_attribute_[$1]], [Check attribute [$1]])
])


#
#  Enable/disable turning on machine-specific optimizations
#
AC_ARG_ENABLE(optimizations,
              AS_HELP_STRING([--enable-optimizations],
                             [Enable non-portable machine-specific CPU optimizations, default: NO]),
              [],
              [enable_optimizations=no])


#
# Check if compiler supports a given CPU optimization flag, and if yes - add it
# to BASE_CFLAGS substitution, and OPT_CFLAGS C define.
#
# Usage: COMPILER_CPU_OPTIMIZATION([name], [doc], [flag], [program])
#
AC_DEFUN([COMPILER_CPU_OPTIMIZATION],
[
    AC_ARG_WITH([$1],
                [AS_HELP_STRING([--with-$1], [Use $2 compiler option.])],
                [],
                [with_$1=$enable_optimizations])
   
    AS_IF([test "x$with_$1" != "xno"],
          [SAVE_CFLAGS="$CFLAGS"
           CFLAGS="$BASE_CFLAGS $CFLAGS $3"
           AC_MSG_CHECKING([$3])
           CHECK_CROSS_COMP([AC_LANG_SOURCE([$4])],
                            [AC_MSG_RESULT([yes])
                             # TODO: Add CPU UARCH detector and validator in UCX init.
                             # As for now we will avoid passing this information to
                             # library.
                             BASE_CFLAGS="$BASE_CFLAGS $3"
                             AS_IF([test "x$1" != "xmcpu" -a "x$1" != "xmarch"],
                                   [OPT_CFLAGS="$OPT_CFLAGS|$1"])],
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
      0x48) case $cpupart in
        0xd01 | 0x0d01)
          AC_DEFINE([HAVE_AARCH64_HI1620], 1, [Huawei Kunpeng 920])
          ax_cpu="tsv110"
          ax_arch="armv8.2-a" ;;
        esac
        ;;
      *)
        ;;
    esac 
    AM_CONDITIONAL([HAVE_AARCH64_THUNDERX2], [test x$ax_cpu = xthunderx2t99])
    AM_CONDITIONAL([HAVE_AARCH64_THUNDERX1], [test x$ax_cpu = xthunderxt88])
    AM_CONDITIONAL([HAVE_AARCH64_HI1620], [test x$ax_cpu = xtsv110])
])


#
# CHECK_COMPILER_FLAG
# Usage: CHECK_COMPILER_FLAG([name], [flag], [program], [if-true], [if-false])
#
# The macro checks if program may be compiled and linked using specified flag
#
AC_DEFUN([CHECK_COMPILER_FLAG],
[
         AC_MSG_CHECKING([compiler flag $1])
         SAVE_CFLAGS="$CFLAGS"
         SAVE_CXXFLAGS="$CFLAGS"
         CFLAGS="$BASE_CFLAGS $CFLAGS $2"
         CXXFLAGS="$BASE_CXXFLAGS $CXXFLAGS $2"
         AC_LINK_IFELSE([$3],
                        [AC_MSG_RESULT([yes])
                         CFLAGS="$SAVE_CFLAGS"
                         CXXFLAGS="$SAVE_CXXFLAGS"
                         $4],
                        [AC_MSG_RESULT([no])
                         CFLAGS="$SAVE_CFLAGS"
                         CXXFLAGS="$SAVE_CXXFLAGS"
                         $5])
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
                             [BASE_CFLAGS="$BASE_CFLAGS $2"
                              $4],
                             [$5])
])


#
# ADD_COMPILER_FLAGS_IF_SUPPORTED
# Usage: ADD_COMPILER_FLAGS_IF_SUPPORTED([[flag1], [flag2], [flag3]], [program])
#
# The macro checks multiple flags supported by compiler
#
AC_DEFUN([ADD_COMPILER_FLAGS_IF_SUPPORTED],
[
         m4_foreach([_flag], [$1],
                    [ADD_COMPILER_FLAG_IF_SUPPORTED([_flag], [_flag], [$2], [], [])])
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
                                  int main(int argc, char** argv) { return f(); }
                            ]])],
                           [AC_MSG_RESULT([yes])
                            $2="${$2} $1"],
                           [AC_MSG_RESULT([no])])
         CFLAGS="$SAVE_CFLAGS"
])


#
# Force ICC treat command line warnings as errors.
# This evaluation should be called prior to all other compiler flags evals
#
ADD_COMPILER_FLAGS_IF_SUPPORTED([[-diag-error 10006],
                                 [-diag-error 10148]],
                                [AC_LANG_SOURCE([[int main(int argc, char **argv){return 0;}]])])


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
                                                 int main(int argc, char** argv) {
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
# which requires alignment of memory buffer by 32 or higer bytes. Default malloc method
# guarantee alignment for 16 bytes only. Force using compiler 16-bytes alignment
# by default if option is supported.
#
UCX_ALLOC_ALIGN=16
ADD_COMPILER_FLAG_IF_SUPPORTED([-fmax-type-align=$UCX_ALLOC_ALIGN],
                               [-fmax-type-align=$UCX_ALLOC_ALIGN],
                               [AC_LANG_SOURCE([[int main(int argc, char** argv){return 0;}]])],
                               [AC_DEFINE_UNQUOTED([UCX_ALLOC_ALIGN], $UCX_ALLOC_ALIGN, [Set alignment assumption for compiler])],
                               [])


#
# SSE/AVX
#
COMPILER_CPU_OPTIMIZATION([avx], [AVX], [-mavx],
                          [#include <immintrin.h>
                           int main(int argc, char** argv) {
                               return _mm256_testz_si256(_mm256_set1_epi32(1), _mm256_set1_epi32(3));
                           }
                          ])
AS_IF([test "x$with_avx" != xyes],
      [COMPILER_CPU_OPTIMIZATION([sse41], [SSE 4.1], [-msse4.1],
                                 [#include <smmintrin.h>
                                  int main(int argc, char** argv) {
                                      return _mm_testz_si128(_mm_set1_epi32(1), _mm_set1_epi32(3));
                                  }
                                 ])
       COMPILER_CPU_OPTIMIZATION([sse42], [SSE 4.2], [-msse4.2],
                                 [#include <popcntintrin.h>
                                  int main(int argc, char** argv) { return _mm_popcnt_u32(0x101) - 2;
                                  }])
      ])


DETECT_UARCH()


#
# CPU tuning
#
AS_IF([test "x$ax_cpu" != "x"],
      [COMPILER_CPU_OPTIMIZATION([mcpu], [CPU Model], [-mcpu=$ax_cpu],
                                 [int main(int argc, char** argv) { return 0;}])
      ])


# 
# Architecture tuning
# 
AS_IF([test "x$ax_arch" != "x"],
      [COMPILER_CPU_OPTIMIZATION([march], [architecture tuning], [-march=$ax_arch],
                                 [int main(int argc, char** argv) { return 0;}])
      ])


#
# Check for compiler attribute which disables optimizations per-function.
#
CHECK_SPECIFIC_ATTRIBUTE([optimize], [NOOPTIMIZE],
                         [int foo (int arg) __attribute__ ((optimize("O0")));])


#
# Compile code with frame pointer. Optimizations usually omit the frame pointer,
# but if we are profiling the code with callgraph we need it.
# This option may affect perofrmance so it is off by default.
#
AC_ARG_ENABLE([frame-pointer],
    AS_HELP_STRING([--enable-frame-pointer],
                   [Compile with frame pointer, useful for profiling, default: NO]),
    [],
    [enable_frame_pointer=no])
AS_IF([test "x$enable_frame_pointer" = xyes -o "x$enable_gcov" = xyes],
      [ADD_COMPILER_FLAG_IF_SUPPORTED([-fno-omit-frame-pointer],
                                      [-fno-omit-frame-pointer],
                                      [AC_LANG_SOURCE([[int main(int argc, char** argv){return 0;}]])],
                                      [AS_MESSAGE([compiling with frame pointer])],
                                      [AS_MESSAGE([compiling with frame pointer is not supported])])],
      [:])

ADD_COMPILER_FLAG_IF_SUPPORTED([-funwind-tables],
                               [-funwind-tables],
                               [AC_LANG_SOURCE([[int main(int argc, char** argv){return 0;}]])],
                               [AS_MESSAGE([compiling with unwind tables])],
                               [AS_MESSAGE([compiling without unwind tables])])


AS_IF([test "x$enable_gcov" = xyes],
      [ADD_COMPILER_FLAGS_IF_SUPPORTED([[-ftest-coverage],
                                        [-fprofile-arcs]],
                                       [AC_LANG_SOURCE([[int main(int argc, char** argv){return 0;}]])])],
      [:])


#
# Check for C++ support
#
CHECK_CXX_COMP()


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
					int main(int argc, char** argv) {
						std::to_string(1);
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
# Check for GNU++11 support
#
AC_MSG_CHECKING([gnu++11 support])
AC_LANG_PUSH([C++])

SAVE_CXXFLAGS="$CXXFLAGS"
CXX11FLAGS="-std=gnu++11"
CXXFLAGS="$CXXFLAGS $CXX11FLAGS"
AC_COMPILE_IFELSE([AC_LANG_SOURCE([[#include <iostream>
					#include <string>
					int main(int argc, char** argv) {
						int a;
						typeof(a) b = 0;
						std::to_string(1);
						return 0;
					} ]])],
                  [AC_MSG_RESULT([yes])
                   AC_SUBST([CXX11FLAGS])
                   gnuxx11_happy=yes],
                  [AC_MSG_RESULT([no])
                   gnuxx11_happy=no])
CXXFLAGS="$SAVE_CXXFLAGS"
AM_CONDITIONAL([HAVE_GNUXX11], [test "x$gnuxx11_happy" != xno])

AC_CHECK_DECL(_GLIBCXX_NOTHROW, have_glibcxx_nothrow=yes,
              have_glibcxx_nothrow=no, [[#include <exception>]])
AM_CONDITIONAL([HAVE_GLIBCXX_NOTHROW], [test "x$have_glibcxx_nothrow" = xyes])

AC_LANG_POP


#
# PGI specific switches
#
# --diag_suppress 1    - Suppress last line ends without a newline
# --diag_suppress 68   - Suppress integer conversion resulted in a change of sign
# --diag_suppress 111  - Suppress statement is unreachable
# --diag_suppress 167  - Suppress int* incompatible with unsigned int*
# --diag_suppress 181  - Suppress incorrect printf format for PGI18 compiler. TODO: remove it after compiler fix
# --diag_suppress 188  - Suppress enumerated type mixed with another type
# --diag_suppress 381  - Suppress extra ";" ignored
# --diag_suppress 1215 - Suppress deprecated API warning for PGI18 compiler
# --diag_suppress 1901 - Use of a const variable in a constant expression is nonstandard in C
# --diag_suppress 1902 - Use of a const variable in a constant expression is nonstandard in C (same as 1901)
ADD_COMPILER_FLAGS_IF_SUPPORTED([[--display_error_number],
                                 [--diag_suppress 1],
                                 [--diag_suppress 68],
                                 [--diag_suppress 111],
                                 [--diag_suppress 167],
                                 [--diag_suppress 181],
                                 [--diag_suppress 188],
                                 [--diag_suppress 381],
                                 [--diag_suppress 1215],
                                 [--diag_suppress 1901],
                                 [--diag_suppress 1902]],
                                [AC_LANG_SOURCE([[int main(int argc, char **argv){return 0;}]])])


#
# Check if "-pedantic" flag is supported
#
CHECK_COMPILER_FLAG([-pedantic], [-pedantic],
                    [AC_LANG_SOURCE([[int main(int argc, char** argv){return 0;}]])],
                    [CFLAGS_PEDANTIC="$CFLAGS_PEDANTIC -pedantic"],
                    [])

#
# Check if "-dynamic-list-data" flag is supported
#
CHECK_COMPILER_FLAG([-Wl,-dynamic-list-data], [-Wl,-dynamic-list-data],
                    [AC_LANG_SOURCE([[int main(int argc, char** argv){return 0;}]])],
                    [LDFLAGS_DYNAMIC_LIST_DATA="-Wl,-dynamic-list-data"],
                    [LDFLAGS_DYNAMIC_LIST_DATA="-Wl,-export-dynamic"])

#
# Add strict compilation flags
#
ADD_COMPILER_FLAGS_IF_SUPPORTED([[-Wno-missing-field-initializers],
                                 [-Wno-unused-parameter],
                                 [-Wno-unused-label],
                                 [-Wno-long-long],
                                 [-Wno-endif-labels],
                                 [-Wno-sign-compare],
                                 [-Wno-multichar],
                                 [-Wno-deprecated-declarations],
                                 [-Winvalid-pch]],
                                [AC_LANG_SOURCE([[int main(int argc, char **argv){return 0;}]])])


#
# Set C++ optimization/debug flags to be the same as for C
#
BASE_CXXFLAGS="$BASE_CFLAGS"


#
# Add strict flags supported by C compiler only
# NOTE: This must be done after setting BASE_CXXFLAGS
#
ADD_COMPILER_FLAGS_IF_SUPPORTED([[-Wno-pointer-sign],
                                 [-Werror-implicit-function-declaration],
                                 [-Wno-format-zero-length],
                                 [-Wnested-externs],
                                 [-Wshadow],
                                 [-Werror=declaration-after-statement]],
                                [AC_LANG_SOURCE([[int main(int argc, char **argv){return 0;}]])])


AC_SUBST([BASE_CFLAGS])
AC_SUBST([BASE_CXXFLAGS])
AC_SUBST([CFLAGS_PEDANTIC])
AC_SUBST([LDFLAGS_DYNAMIC_LIST_DATA])


#
# Set common C preprocessor flags
#
BASE_CPPFLAGS="-DCPU_FLAGS=\"$OPT_CFLAGS\""
BASE_CPPFLAGS="$BASE_CPPFLAGS -I\${abs_top_srcdir}/src"
BASE_CPPFLAGS="$BASE_CPPFLAGS -I\${abs_top_builddir}"
BASE_CPPFLAGS="$BASE_CPPFLAGS -I\${abs_top_builddir}/src"
AC_SUBST([BASE_CPPFLAGS], [$BASE_CPPFLAGS])
