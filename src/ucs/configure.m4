#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
# Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
# Copyright (C) ARM, Ltd. 2016. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

ucs_modules=""
m4_include([src/ucs/vfs/sock/configure.m4])
m4_include([src/ucs/vfs/fuse/configure.m4])
AC_DEFINE_UNQUOTED([ucs_MODULES], ["${ucs_modules}"], [UCS loadable modules])

#
# Internal profiling support.
# This option may affect perofrmance so it is off by default.
#
AC_ARG_ENABLE([profiling],
	AS_HELP_STRING([--enable-profiling], [Enable profiling support, default: NO]),
	[],
	[enable_profiling=no])

AS_IF([test "x$enable_profiling" = xyes],
	[AS_MESSAGE([enabling profiling])
	 AC_DEFINE([HAVE_PROFILING], [1], [Enable profiling])
	 HAVE_PROFILING=yes]
	[:]
)
AM_CONDITIONAL([HAVE_PROFILING],[test "x$HAVE_PROFILING" = "xyes"])


#
# Detailed backtrace with debug information.
# This option requires binutils-devel package.
#
AC_ARG_WITH([bfd],
            [AS_HELP_STRING([--with-bfd=(DIR)],
            [Enable using BFD support for detailed backtrace (default is guess).])],
            [], [with_bfd=guess])
AS_IF([test "x$with_bfd" != xno],
      [
       # Do not define BFD_CFLAGS, BFD_LIBS, etc to make sure automake will not
       # try to use them when bfd_happy=no
       BFD_CHECK_CFLAGS=""
       BFD_CHECK_LIBS="-lbfd -ldl -lz"
       AS_IF([test "x$with_bfd" = "xguess" -o "x$with_bfd" = "xyes"],
             [BFD_CHECK_CPPFLAGS=""
              BFD_CHECK_LDFLAGS=""],
             [BFD_CHECK_CPPFLAGS="-I${with_bfd}/include"
              BFD_CHECK_LDFLAGS="-L${with_bfd}/lib -L${with_bfd}/lib64"])

       save_CFLAGS="$CFLAGS"
       save_CPPFLAGS="$CPPFLAGS"
       save_LDFLAGS="$LDFLAGS"
       save_LIBS="$LIBS"

       # Check BFD properties with all flags pointing to the custom location
       CPPFLAGS="$CPPFLAGS $BFD_CHECK_CPPFLAGS"
       LIBS="$LIBS $BFD_CHECK_LIBS"
       BFD_CHECK_DEPLIBS="-liberty -lz -ldl"

       # Link the test applications as a shared library, to fail if libbfd is
       # not a PIC object.
       # Do not allow undefined symbols, to ensure all references are resolved.
       # TODO Allow static link with static libbfd
       CFLAGS="$CFLAGS $BFD_CHECK_CFLAGS -fPIC"
       LDFLAGS="$LDFLAGS $BFD_CHECK_LDFLAGS -shared -Wl,--no-undefined"

       bfd_happy="yes"
       AC_CHECK_LIB(bfd, bfd_openr, [],
                    [
                     # If cannot link with bfd, try adding known dependency libs
                     # unset the cached check result to force re-check
                     unset ac_cv_lib_bfd_bfd_openr
                     AC_CHECK_LIB(bfd, bfd_openr,
                                  [BFD_CHECK_LIBS="$BFD_CHECK_LIBS $BFD_CHECK_DEPLIBS"
                                   LIBS="$LIBS $BFD_CHECK_DEPLIBS"],
                                  [bfd_happy="no"],
                                  [$BFD_CHECK_DEPLIBS])
                    ])
       AC_CHECK_HEADER([bfd.h], [], [bfd_happy="no"])
       AC_CHECK_TYPES([struct dl_phdr_info], [], [bfd_happy=no],
                      [[#define _GNU_SOURCE 1
                        #include <link.h>]])

       AS_IF([test "x$bfd_happy" = "xyes"],
             [
              # Check optional BFD functions
              AC_CHECK_DECLS([bfd_get_section_flags, bfd_section_flags,
                              bfd_get_section_vma, bfd_section_vma],
                             [], [], [#include <bfd.h>])

              # Check bfd_section_size() function type
              AC_MSG_CHECKING([bfd_section_size API version])
              AC_LANG_PUSH([C])
              AC_COMPILE_IFELSE([
                  AC_LANG_SOURCE([[
                      #include <bfd.h>
                      int main(int argc, char** argv) {
                          asection sec;
                          bfd_section_size(&sec);
                          return 0;
                      }
                  ]])],
                  [AC_MSG_RESULT([1-arg API])
                   AC_DEFINE([HAVE_1_ARG_BFD_SECTION_SIZE], [1], [bfd_section_size 1-arg])],
                  [AC_MSG_RESULT([2-args API])
                   AC_DEFINE([HAVE_1_ARG_BFD_SECTION_SIZE], [0], [bfd_section_size 2-args])
              ])
              AC_LANG_POP([C])

              # Check if demange is supported
              AC_CHECK_FUNCS([cplus_demangle])

              case ${host} in
                  aarch64*) BFD_CHECK_CFLAGS="$BFD_CHECK_CFLAGS -funwind-tables" ;;
              esac

              # Define macros and variable substitutions for BFD support
              AC_DEFINE([HAVE_DETAILED_BACKTRACE], 1, [Enable detailed backtrace])
              AC_SUBST([BFD_CFLAGS], [$BFD_CHECK_CFLAGS])
              AC_SUBST([BFD_CPPFLAGS], [$BFD_CHECK_CPPFLAGS])
              AC_SUBST([BFD_LIBS], [$BFD_CHECK_LIBS])
              AC_SUBST([BFD_LDFLAGS], [$BFD_CHECK_LDFLAGS])
              AC_SUBST([BFD_DEPS], [$BFD_CHECK_DEPLIBS])
             ],
             [
               AS_IF([test "x$with_bfd" != "xyes" -a "x$with_bfd" != "xguess"],
                     [AC_MSG_ERROR([BFD support requested but could not be found])])
             ])

       LIBS="$save_LIBS"
       LDFLAGS="$save_LDFLAGS"
       CPPFLAGS="$save_CPPFLAGS"
       CFLAGS="$save_CFLAGS"
      ],
      [bfd_happy="no"
       AC_MSG_WARN([BFD support was explicitly disabled])]
)


#
# Enable statistics and counters
#
AC_ARG_ENABLE([stats],
	AS_HELP_STRING([--enable-stats],
	               [Enable statistics, useful for profiling, default: NO]),
	[],
	[enable_stats=no])

AS_IF([test "x$enable_stats" = xyes],
	  [AS_MESSAGE([enabling statistics])
	   AC_DEFINE([ENABLE_STATS], [1], [Enable statistics])
	   HAVE_STATS=yes],
	  [:]
  )
AM_CONDITIONAL([HAVE_STATS],[test "x$HAVE_STATS" = "xyes"])


#
# Enable tuning params at runtime
#
AC_ARG_ENABLE([tuning],
	AS_HELP_STRING([--enable-tuning],
	               [Enable parameter tuning in run-time, default: NO]),
	[],
	[enable_tuning=no])

AS_IF([test "x$enable_tuning" = xyes],
	  [AS_MESSAGE([enabling tuning])
	   AC_DEFINE([ENABLE_TUNING], [1], [Enable tuning])
	   HAVE_TUNING=yes],
	  [:]
  )
AM_CONDITIONAL([HAVE_TUNING],[test "x$HAVE_TUNING" = "xyes"])


#
# Disable logging levels below INFO
#
AC_ARG_ENABLE([logging],
	AS_HELP_STRING([--enable-logging],
	               [Enable debug logging, default: YES])
	)

AS_CASE([$enable_logging],
        [no],          [AC_DEFINE([UCS_MAX_LOG_LEVEL], [UCS_LOG_LEVEL_DEBUG], [Highest log level])],
        [warn],        [AC_DEFINE([UCS_MAX_LOG_LEVEL], [UCS_LOG_LEVEL_WARN], [Highest log level])],
        [diag],        [AC_DEFINE([UCS_MAX_LOG_LEVEL], [UCS_LOG_LEVEL_DIAG], [Highest log level])],
        [info],        [AC_DEFINE([UCS_MAX_LOG_LEVEL], [UCS_LOG_LEVEL_INFO], [Highest log level])],
        [debug],       [AC_DEFINE([UCS_MAX_LOG_LEVEL], [UCS_LOG_LEVEL_DEBUG], [Highest log level])],
        [trace],       [AC_DEFINE([UCS_MAX_LOG_LEVEL], [UCS_LOG_LEVEL_TRACE], [Highest log level])],
        [trace_req],   [AC_DEFINE([UCS_MAX_LOG_LEVEL], [UCS_LOG_LEVEL_TRACE_REQ], [Highest log level])],
        [trace_data],  [AC_DEFINE([UCS_MAX_LOG_LEVEL], [UCS_LOG_LEVEL_TRACE_DATA], [Highest log level])],
        [trace_async], [AC_DEFINE([UCS_MAX_LOG_LEVEL], [UCS_LOG_LEVEL_TRACE_ASYNC], [Highest log level])],
        [trace_func],  [AC_DEFINE([UCS_MAX_LOG_LEVEL], [UCS_LOG_LEVEL_TRACE_FUNC], [Highest log level])],
        [trace_poll],  [AC_DEFINE([UCS_MAX_LOG_LEVEL], [UCS_LOG_LEVEL_TRACE_POLL], [Highest log level])],
                       [AC_DEFINE([UCS_MAX_LOG_LEVEL], [UCS_LOG_LEVEL_TRACE_POLL], [Highest log level])])

#
# Disable assertions
#
AC_ARG_ENABLE([assertions],
	AS_HELP_STRING([--disable-assertions],
	               [Disable code assertions, default: NO])
	)

AS_IF([test "x$enable_assertions" != xno],
		AC_DEFINE([ENABLE_ASSERT], [1], [Enable assertions])
	)

#
# Check if __attribute__((constructor)) works
#
AC_MSG_CHECKING([__attribute__((constructor))])
CHECK_CROSS_COMP([AC_LANG_SOURCE([static int rc = 1;
                  static void constructor_test() __attribute__((constructor));
                  static void constructor_test() { rc = 0; }
                  int main(int argc, char** argv) { return rc; }])],
                [AC_MSG_RESULT([yes])],
                [AC_MSG_ERROR([Cannot continue. Please use compiler that
                             supports __attribute__((constructor))])]
                )


#
# Manual configuration of cacheline size
#
AC_ARG_WITH([cache-line-size],
        [AC_HELP_STRING([--with-cache-line-size=SIZE],
            [Build UCX with cache line size defined by user. This parameter
             overwrites default cache line sizes defines in
             UCX (x86-64: 64, Power: 128, ARMv8: 64/128). The supported values are: 64, 128])],
        [],
        [with_cache_line_size=no])

AS_IF([test "x$with_cache_line_size" != xno],[
	     case ${with_cache_line_size} in
                 64)
		     AC_MSG_RESULT(The cache line size is set to 64B)
		     AC_DEFINE([HAVE_CACHE_LINE_SIZE], 64, [user defined cache line size])
		     ;;
		 128)
		     AC_MSG_RESULT(The cache line size is set to 128B)
		     AC_DEFINE([HAVE_CACHE_LINE_SIZE], 128, [user defined cache line size])
		     ;;
		 @<:@0-9@:>@*)
		     AC_MSG_WARN(Unusual cache cache line size was specified: [$with_cache_line_size])
		     AC_DEFINE_UNQUOTED([HAVE_CACHE_LINE_SIZE], [$with_cache_line_size], [user defined cache line size])
		     ;;
		 *)
		     AC_MSG_ERROR(Cannot continue. Unsupported cache line size [$with_cache_line_size].)
		     ;;
             esac],
	     [])


#
# Architecture specific checks
#
case ${host} in
    aarch64*)
    AC_MSG_CHECKING([support for CNTVCT_EL0 on aarch64])
    AC_RUN_IFELSE([AC_LANG_PROGRAM([[#include <stdint.h>]],
                                   [[uint64_t tmp; asm volatile("mrs %0, cntvct_el0" : "=r" (tmp));
                                   ]])],
                                   [AC_MSG_RESULT([yes])
                                    AC_DEFINE([HAVE_HW_TIMER], [1], [high-resolution hardware timer enabled])],
                                   [AC_MSG_RESULT([no])
                                    AC_DEFINE([HAVE_HW_TIMER], [0], [high-resolution hardware timer disabled])],
                                   [AC_MSG_RESULT([no - cross-compiling detected])
                                    AC_DEFINE([HAVE_HW_TIMER], [0], [high-resolution hardware timer disabled])]
                  );;
    *)
    # HW timer is supported for all other architectures
    AC_DEFINE([HAVE_HW_TIMER], [1], [high-resolution hardware timer enabled])
esac

#
# Enable built-in memcpy
#
AC_ARG_ENABLE([builtin-memcpy],
	AS_HELP_STRING([--enable-builtin-memcpy],
	               [Enable builtin memcpy routine, default: YES]),
	[],
	[enable_builtin_memcpy=yes])

AS_IF([test "x$enable_builtin_memcpy" != xno],
	  [AS_MESSAGE([enabling builtin memcpy])
	   AC_DEFINE([ENABLE_BUILTIN_MEMCPY], [1], [Enable builtin memcpy])],
	  [AC_DEFINE([ENABLE_BUILTIN_MEMCPY], [0], [Enable builtin memcpy])]
  )

AC_CHECK_FUNCS([__clear_cache], [], [])
AC_CHECK_FUNCS([__aarch64_sync_cache_range], [], [])


AC_CONFIG_FILES([src/ucs/Makefile
                 src/ucs/signal/Makefile
                 src/ucs/ucx-ucs.pc])
