#
# Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#


#
# Enable overriding library symbols
#
AC_ARG_ENABLE([symbol-override],
	AS_HELP_STRING([--disable-symbol-override], [Disable overriding library symbols, default: NO]),
	[],
	[enable_symbol_override=yes])
	
AS_IF([test "x$enable_symbol_override" == xyes], 
	[AC_DEFINE([ENABLE_SYMBOL_OVERRIDE], [1], [Enable symbol override])]
	[:]
)

#
# Memory allocator selection
#
AC_ARG_WITH([allocator],
    [AC_HELP_STRING([--with-allocator=NAME],
        [Build UCX with predefined memory allocator. The supported values are:
         ptmalloc286. Default: ptmalloc286])],
        [],
        [with_allocator=ptmalloc286])

case ${with_allocator} in
    ptmalloc286)
        AC_MSG_NOTICE(Memory allocator is ptmalloc-2.8.6 version)
        AC_DEFINE([HAVE_UCM_PTMALLOC286], 1, [Use ptmalloc-2.8.6 version])
        HAVE_UCM_PTMALLOC286=yes
        ;;
    *)
        AC_MSG_ERROR(Cannot continue. Unsupported memory allocator name
                     in --with-allocator=[$with_allocator])
        ;;
esac

AM_CONDITIONAL([HAVE_UCM_PTMALLOC286],[test "x$HAVE_UCM_PTMALLOC286" = "xyes"])

AC_CHECK_FUNCS([malloc_get_state malloc_set_state],
               [],
               [],
               [#include <stdlib.h>])

