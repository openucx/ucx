#
# Copyright (C) Mellanox Technologies Ltd. 2001-2011.  ALL RIGHTS RESERVED.
#
# $COPYRIGHT$
# $HEADER$

define([scm_r], esyscmd([sh -c "git rev-parse --short=7 HEAD"]))
define([ucx_ver_major], 0)
define([ucx_ver_minor], 1)
define([ts], esyscmd([sh -c "date +%Y%m%d%H%M%S"]))
define([revcount], esyscmd([git rev-list HEAD | wc -l | sed -e 's/ *//g' | xargs -n1 printf]))

# This is the API version (see libtool library versioning)
# http://www.gnu.org/software/libtool/manual/html_node/Updating-version-info.html
# current:rev:age
define([libucx_so_version], 2:1:0)

AC_INIT([ucx], [ucx_ver_major.ucx_ver_minor])

config_flags="$*"

AC_CONFIG_HEADERS([config.h])
AM_INIT_AUTOMAKE([1.10 foreign tar-ustar silent-rules subdir-objects])
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])
AM_MAINTAINER_MODE
AC_CONFIG_MACRO_DIR([config/m4])

AC_PREFIX_DEFAULT([/usr])

RPM_RELEASE=1
BUILD_TS=ts
MAJOR_VERSION=ucx_ver_major
MINOR_VERSION=ucx_ver_minor
MICRO_VERSION=scm_r
REV_COUNT=revcount
VERSION=$MAJOR_VERSION.$MINOR_VERSION.$REV_COUNT
SOVERSION=libucx_so_version
AC_SUBST(MAJOR_VERSION)
AC_SUBST(MINOR_VERSION)
AC_SUBST(MICRO_VERSION)
AC_SUBST(REV_COUNT)
AC_SUBST(RPM_RELEASE)
AC_SUBST(SOVERSION)
AC_SUBST(BUILD_TS)

AC_PROG_CC
AC_PROG_CXX
AM_PROG_AS
AC_PROG_SED
AC_PROG_INSTALL
AC_PROG_LIBTOOL
AC_HEADER_STDC
LT_LIB_M
AC_CHECK_LIB([rt], [shm_open], [], AC_MSG_ERROR([librt not found]))
AC_SEARCH_LIBS(pthread_create, pthread)
AC_CHECK_DECLS([F_SETOWN_EX], [], [], [#define _GNU_SOURCE 1
#include <fcntl.h>])
AC_CHECK_DECLS([__ppc_get_timebase_freq], [], [], [#include <sys/platform/ppc.h>])
AC_CHECK_HEADERS([sys/platform/ppc.h])

PKG_PROG_PKG_CONFIG
PKG_CONFIG="pkg-config"
if test -n "${host}" ; then
   PKG_CONFIG=${host}-${PKG_CONFIG}
fi


#
# Save config flags for version dump tool
#
AC_DEFINE_UNQUOTED([UCX_CONFIGURE_FLAGS], ["$config_flags"], [UCX configure flags])


#
# Provide the functionality of AS_VAR_APPEND if Autoconf does not have it.
#
m4_ifdef([AS_VAR_APPEND],
[m4_copy([AS_VAR_APPEND], [ucx_AS_VAR_APPEND])],
[m4_define([ucx_AS_VAR_APPEND],
[AS_VAR_SET([$1], [AS_VAR_GET([$1])$2])])])


#
# Initialize CFLAGS
#
CFLAGS="-g -Wall -Werror $UCX_CFLAGS"
CXXFLAGS="$CFLAGS"


#
# Zlib
#
AC_ARG_WITH([zlib],
            [AC_HELP_STRING([--with-zlib=DIR],
                            [Specify path to external zlib library.])],
            [if test "$withval" != no; then
               if test "$withval" != yes; then
                 ZLIB_DIR=$withval
               fi
             fi])
if test -n "$ZLIB_DIR"; then
  LDFLAGS="$LDFLAGS -L$ZLIB_DIR"
fi


#
# RC Support
#
AC_ARG_WITH([rc],
            [AC_HELP_STRING([--with-rc], [Compile with RC support])],
            [],
            [with_rc=yes])
AM_CONDITIONAL([HAVE_TL_RC], [test "x$with_rc" != xno])
AS_IF([test "x$with_rc" != xno], 
      [AC_DEFINE([HAVE_TL_RC], 1, [RC transport support])
       transports="${transports},rc"])


# Print which transports are supported
AC_MSG_NOTICE([Supported transports: $transports])


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
# Valgrind support
#
AC_ARG_WITH([valgrind],
    AC_HELP_STRING([--with-valgrind],
                   [Enable Valgrind annotations (small runtime overhead, default NO)]),
    [],
    [with_valgrind=no]
)
AS_IF([test "x$with_valgrind" == xno],
      [AC_DEFINE([NVALGRIND], 1, [Define to 1 to disable Valgrind annotations.])
      ],
      [AC_CHECK_HEADER([valgrind/memcheck.h], [],
                       [AC_MSG_ERROR([Valgrind memcheck support requested, but <valgrind/memcheck.h> not found, install valgrind-devel rpm.])])
       if test -d $with_valgrind; then
          CPPFLAGS="$CPPFLAGS -I$with_valgrind/include"
       fi
      ]
)


#
# Zlib
#
AC_CHECK_LIB(z, compress2,, AC_MSG_WARN([zlib library not found]))


#
# Google Testing framework
#
GTEST_LIB_CHECK([1.5.0], [true], [true])


#
# Boost C++ library (if we're using gtest)
#
if test "x$HAVE_GTEST" = "xyes"; then
	AX_BOOST_BASE([1.38], [],
	              [AC_MSG_ERROR([Please install boost development libraries])])
	AX_BOOST_THREAD
fi


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
     if test "x$BT" == "x1"; then
         AC_DEFINE([HAVE_DETAILED_BACKTRACE], 1, [Enable detailed backtraces])
     fi
	]
)


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
	
AS_IF([test "x$enable_frame_pointer" == xyes], 
	  [AS_MESSAGE([compiling with frame pointer])
	   CFLAGS="$CFLAGS -fno-omit-frame-pointer"],
	  [:]
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
# Enable fault injection code
#
AC_ARG_ENABLE([fault-injection],
	AS_HELP_STRING([--enable-fault-injection], 
	               [Enable fault injection code, default: NO]),
	[],
	[enable_fault_injection=no])
	
AS_IF([test "x$enable_fault_injection" == xyes], 
	  [AS_MESSAGE([enabling with fault injection code])
	   AC_DEFINE([ENABLE_FAULT_INJECTION], [1], [Enable fault injection code])],
	  [:]
  )


#
# Disable logging levels below INFO
#
AC_ARG_ENABLE([logging],
	AS_HELP_STRING([--enable-logging],
	               [Enable debug logging, default: YES])
	)

AS_IF([test "x$enable_logging" != xno],
        [AC_DEFINE([UCX_MAX_LOG_LEVEL], [UCX_LOG_LEVEL_TRACE_POLL], [Highest log level])],
        [AC_DEFINE([UCX_MAX_LOG_LEVEL], [UCX_LOG_LEVEL_INFO], [Highest log level])]
    )


#
# Disable assertions
#
AC_ARG_ENABLE([assertions],
	AS_HELP_STRING([--disable-assertions], 
	               [Disable code assertions, default: NO]),
	[],
	[AC_DEFINE([ENABLE_ASSERT], [1], [Enable assertions])])


#
# Disable checking user parameters
#
AC_ARG_ENABLE([params-check],
	AS_HELP_STRING([--disable-params-check], 
	               [Disable checking user parameters passed to API, default: NO]),
	[AC_DEFINE([ENABLE_PARAMS_CHECK], [0])],
	[AC_DEFINE([ENABLE_PARAMS_CHECK], [1], [Enable checking user parameters])])


#
# Enable collecting data to ease debugging
#
AC_ARG_ENABLE([debug-data],
	AS_HELP_STRING([--enable-debug-data],
	               [Enable collecting data to ease debugging, default: NO]),
	[AC_DEFINE([ENABLE_DEBUG_DATA], [1], [Enable collecting data])],
	[AC_DEFINE([ENABLE_DEBUG_DATA], [0])])


#
# Final output
#
AC_CONFIG_FILES([
				 Makefile
                 src/ucs/Makefile
                 src/uct/Makefile
                 src/uct/api/version.h
                 ])

AC_OUTPUT
