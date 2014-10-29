#
# Copyright (C) Mellanox Technologies Ltd. 2001-2011.  ALL RIGHTS RESERVED.
#
# $COPYRIGHT$
# $HEADER$
#


#
# Internal instrumentation support.
# This option may affect perofrmance so it is off by default.
#
AC_ARG_ENABLE([instrumentation],
	AS_HELP_STRING([--enable-instrumentation], [Enable instrumentation support, default: NO]),
	[],
	[enable_instrumentation=no])
	
AS_IF([test "x$enable_instrumentation" == xyes], 
	[AC_DEFINE([HAVE_INSTRUMENTATION], [1], [Enable instrumentation])]
	[:]
)


#
# Detailed backtrace with debug information.
# This option requires binutils-devel package.
#
AC_ARG_ENABLE([backtrace-detail],
	AS_HELP_STRING([--disable-backtrace-detail], [Disable detailed backtrace support, default: NO]),
	[],
	[enable_backtrace_detail=yes])
	
AS_IF([test "x$enable_backtrace_detail" == xyes], 
	[
	BT=1
	AC_CHECK_HEADER([bfd.h], [], [AC_MSG_WARN([binutils headers not found])]; BT=0)
	AC_CHECK_HEADERS([libiberty.h libiberty/libiberty.h], [], [],
					[#define HAVE_DECL_BASENAME 1])
	if test "x$ac_cv_header_libiberty_h" == "x" && test "x$ac_cv_header_libiberty_libiberty_h" == "x"; then
	    AC_MSG_WARN([binutils headers not found]); BT=0
	fi
	AC_CHECK_LIB(bfd, bfd_init,  LIBS="$LIBS -lbfd", [AC_MSG_WARN([bfd library not found])];BT=0)
	AC_CHECK_LIB(iberty, xstrerror, LIBS="$LIBS -liberty", [AC_MSG_WARN([iberty library not found])];BT=0)
	AC_CHECK_LIB(dl, dlopen, LIBS="$LIBS -ldl", [AC_MSG_WARN([dl library not found])];BT=0)
	AC_CHECK_LIB(intl, main, LIBS="$LIBS -lintl", [AC_MSG_WARN([intl library not found])])
	AC_CHECK_TYPES([struct dl_phdr_info], [], [AC_MSG_WARN([struct dl_phdr_info not defined])];BT=0,
					[#define _GNU_SOURCE 1
					 #include <link.h>]) 
	if test "x$BT" == "x1"; then
		AC_DEFINE([HAVE_DETAILED_BACKTRACE], 1, [Enable detailed backtrace])
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
	
AS_IF([test "x$enable_stats" == xyes], 
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
	
AS_IF([test "x$enable_tuning" == xyes], 
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
	
AS_IF([test "x$enable_memtrack" == xyes], 
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
	               [Disable code assertions, default: NO]),
	[],
	[AC_DEFINE([ENABLE_ASSERT], [1], [Enable assertions])])


