#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
# Copyright (C) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
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
      [AC_DEFINE([NVALGRIND], 1, [Define to 1 to disable Valgrind annotations.])],
      [AS_IF([test ! -d $with_valgrind], 
              [AC_MSG_NOTICE([Valgrind path was not defined, guessing ...])
               with_valgrind=/usr], [:])
        AC_CHECK_HEADER([$with_valgrind/include/valgrind/memcheck.h], [],
                       [AC_MSG_ERROR([Valgrind memcheck support requested, but <valgrind/memcheck.h> not found, install valgrind-devel rpm.])])
        CPPFLAGS="$CPPFLAGS -I$with_valgrind/include"
      ]
)


#
# Malloc hooks
#
AC_MSG_CHECKING([malloc hooks])
SAVE_CFLAGS=$CFLAGS
CFLAGS="$CFLAGS $CFLAGS_NO_DEPRECATED"
CHECK_CROSS_COMP([AC_LANG_SOURCE([#include <malloc.h>
                                  static int rc = 1;
                                  void *ptr;
                                  void *myhook(size_t size, const void *caller) {
                                      rc = 0;
                                      return NULL;
                                  }
                                  int main() {
                                      __malloc_hook = myhook;
                                      ptr = malloc(1);
                                      return rc;
                                  }])],
                 [AC_MSG_RESULT([yes])
                  AC_DEFINE([HAVE_MALLOC_HOOK], 1, [malloc hooks support])],
                 [AC_MSG_RESULT([no])
                  AC_MSG_WARN([malloc hooks are not supported])]
                )
CFLAGS=$SAVE_CFLAGS
