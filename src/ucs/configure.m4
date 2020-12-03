#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
# Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
# Copyright (C) ARM, Ltd. 2016. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#


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
AC_ARG_ENABLE([backtrace-detail],
	AS_HELP_STRING([--disable-backtrace-detail], [Disable detailed backtrace support, default: NO]),
	[],
	[enable_backtrace_detail=yes])
	
AS_IF([test "x$enable_backtrace_detail" = xyes],
	[
	BT=1
	AC_CHECK_HEADER([bfd.h], [], [AC_MSG_WARN([binutils headers not found])]; BT=0)
	AC_CHECK_LIB(bfd, bfd_openr,  LIBS="$LIBS -lbfd", [AC_MSG_WARN([bfd library not found])];BT=0)
	AC_CHECK_LIB(dl, dlopen, LIBS="$LIBS -ldl", [AC_MSG_WARN([dl library not found])];BT=0)
	AC_CHECK_LIB(intl, main, LIBS="$LIBS -lintl", [AC_MSG_WARN([intl library not found])])
	AC_CHECK_TYPES([struct dl_phdr_info], [], [AC_MSG_WARN([struct dl_phdr_info not defined])];BT=0,
					[#define _GNU_SOURCE 1
					 #include <link.h>]) 
	AC_CHECK_DECLS([bfd_get_section_flags, bfd_section_flags, bfd_get_section_vma, bfd_section_vma],
		       [], [], [#include <bfd.h>])

	AC_MSG_CHECKING([bfd_section_size API version])
	AC_LANG_PUSH([C])
	SAVE_CFLAGS="$CFLAGS"
	AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
		#include <bfd.h>
		int main(int argc, char** argv) {
			asection *sec = malloc(sizeof(*sec));
			bfd_section_size(sec);
			free(sec);
			return 0;
		} ]])],
		[AC_MSG_RESULT([1-arg API])
		 AC_DEFINE([HAVE_1_ARG_BFD_SECTION_SIZE], [1],
			   [bfd_section_size 1-arg version])],
		[AC_MSG_RESULT([2-args API])
		 AC_DEFINE([HAVE_1_ARG_BFD_SECTION_SIZE], [0],
			   [bfd_section_size 2-args version])])
	CFLAGS="$SAVE_CFLAGS"
	AC_LANG_POP([C])

	if test "x$BT" = "x1"; then
		AC_CHECK_FUNCS([cplus_demangle])
		AC_DEFINE([HAVE_DETAILED_BACKTRACE], 1, [Enable detailed backtrace])
        case ${host} in
            aarch64*) CFLAGS="$CFLAGS -funwind-tables" ;;
        esac
	else
		AC_MSG_WARN([detailed backtrace is not supported])
	fi
	]
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
# Enable memory tracking
#
AC_ARG_ENABLE([memtrack],
	AS_HELP_STRING([--enable-memtrack], 
	               [Enable memory tracking, useful for profiling, default: NO]),
	[],
	[enable_memtrack=no])
	
AS_IF([test "x$enable_memtrack" = xyes],
	  [AS_MESSAGE([enabling memory tracking])
	   AC_DEFINE([ENABLE_MEMTRACK], [1], [Enable memory tracking])
	   HAVE_MEMTRACK=yes],
	  [:]
  )
AM_CONDITIONAL([HAVE_MEMTRACK],[test "x$HAVE_MEMTRACK" = "xyes"])


#
# Disable logging levels below INFO
#
AC_ARG_ENABLE([logging],
	AS_HELP_STRING([--enable-logging],
	               [Enable debug logging, default: YES])
	)

AS_IF([test "x$enable_logging" != xno],
        [AC_DEFINE([UCS_MAX_LOG_LEVEL], [UCS_LOG_LEVEL_TRACE_POLL], [Highest log level])],
        [AC_DEFINE([UCS_MAX_LOG_LEVEL], [UCS_LOG_LEVEL_INFO], [Highest log level])]
    )

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
    AC_RUN_IFELSE([AC_LANG_PROGRAM(
                  [[#include <stdint.h>]],
                  [[uint64_t tmp; asm volatile("mrs %0, cntvct_el0" : "=r" (tmp));]])],
                  [AC_MSG_RESULT([yes])]
		  [AC_DEFINE([HAVE_HW_TIMER], [1], [high-resolution hardware timer enabled])],
		  [AC_MSG_RESULT([no])]
		  [AC_DEFINE([HAVE_HW_TIMER], [0], [high-resolution hardware timer disabled])]
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
