#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
# Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
# $COPYRIGHT$
# $HEADER$
#


#
# SystemV shared memory
#
#IPC_INFO
AC_CHECK_LIB([rt], [shm_open],     [], AC_MSG_ERROR([librt not found]))
AC_CHECK_LIB([rt], [timer_create], [], AC_MSG_ERROR([librt not found]))

#
# Extended string functions
#
AC_CHECK_DECLS([asprintf, strdupa, basename, fmemopen], [],
				AC_MSG_ERROR([GNU string extensions not found]), 
				[#define _GNU_SOURCE 1
				 #include <string.h>
				 #include <stdio.h>])


#
# CPU-sets 
#
AC_CHECK_DECLS([CPU_ZERO, CPU_ISSET], [], 
				AC_MSG_ERROR([CPU_ZERO/CPU_ISSET not found]), 
				[#define _GNU_SOURCE 1
				 #include <sched.h>])


#
# pthread
#
AC_SEARCH_LIBS(pthread_create, pthread)


#
# Route file descriptor signal to specific thread
#
AC_CHECK_DECLS([F_SETOWN_EX], [], [], [#define _GNU_SOURCE 1
#include <fcntl.h>])


#
# PowerPC query for TB frequency
#
AC_CHECK_DECLS([__ppc_get_timebase_freq], [], [], [#include <sys/platform/ppc.h>])
AC_CHECK_HEADERS([sys/platform/ppc.h])


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
AC_ARG_WITH([boost],
    AC_HELP_STRING([--with-boost],
                   [Enable Boost C++ library (required by gtest)]),
    [],
    [with_boost=no])

if test "x$HAVE_GTEST" = "xyes"; then

    AC_LANG_PUSH([C++])
    AC_MSG_CHECKING([for boost])

    AS_IF([test "x$with_boost" != xno],
      [GTEST_CPPFLAGS="$GTEST_CPPFLAGS -I$with_boost"
       GTEST_LDFLAGS="$GTEST_LDFLAGS -L$with_boost/stage/lib"],[])
      
    ORIG_CXXFLAGS=$CXXFLAGS
    CXXFLAGS="$GTEST_CPPFLAGS $CXXFLAGS"

	AC_COMPILE_IFELSE(
		[AC_LANG_PROGRAM([[#include <boost/version.hpp>]]
					 [[
					 #if (BOOST_VERSION < 103800)
					 #  error Failed
					 #endif
					 ]])],
					AC_MSG_RESULT([yes]),
					AC_MSG_ERROR([Please install boost development libraries version 1.38 of above]))

	AC_CHECK_DECLS([BOOST_FOREACH], [], AC_MSG_ERROR([BOOST_FOREACH not supported]),
	               [#include <boost/foreach.hpp>])

    CXXFLAGS=$ORIG_CXXFLAGS
	AC_LANG_POP
fi


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

