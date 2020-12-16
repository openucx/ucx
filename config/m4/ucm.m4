#
# Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#


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


#
# Madvise flags
#
AC_CHECK_DECLS([MADV_FREE,
                MADV_REMOVE,
                POSIX_MADV_DONTNEED],
               [],
               [],
               [#include <sys/mman.h>])


# BISTRO hooks infrastructure
#
# SYS_xxx macro
#
mmap_hooks_happy=yes
AC_CHECK_DECLS([SYS_mmap,
                SYS_munmap,
                SYS_mremap,
                SYS_brk,
                SYS_madvise],
               [],
               [mmap_hooks_happy=no], dnl mmap syscalls are not defined
               [#include <sys/syscall.h>])

shm_hooks_happy=yes
AC_CHECK_DECLS([SYS_shmat,
                SYS_shmdt],
               [],
               [shm_hooks_happy=no],
               [#include <sys/syscall.h>])

ipc_hooks_happy=yes
AC_CHECK_DECLS([SYS_ipc],
               [],
               [ipc_hooks_happy=no],
               [#include <sys/syscall.h>])

AS_IF([test "x$mmap_hooks_happy" = "xyes"],
      AS_IF([test "x$ipc_hooks_happy" = "xyes" -o "x$shm_hooks_happy" = "xyes"],
            [bistro_hooks_happy=yes]))

AS_IF([test "x$bistro_hooks_happy" = "xyes"],
      [AC_DEFINE([UCM_BISTRO_HOOKS], [1], [Enable BISTRO hooks])],
      [AC_DEFINE([UCM_BISTRO_HOOKS], [0], [Enable BISTRO hooks])
       AC_MSG_WARN([Some of required syscalls could not be found])
       AC_MSG_WARN([BISTRO mmap hook mode is disabled])])

AC_CHECK_FUNCS([__curbrk], [], [], [])

#
# tcmalloc library - for testing only
#
SAVE_LDFLAGS="$LDFLAGS"
AC_CHECK_LIB([tcmalloc], [tc_malloc],
             [have_tcmalloc=yes
              TCMALLOC_LIB="-ltcmalloc"],
             [have_tcmalloc=no])
AM_CONDITIONAL([HAVE_TCMALLOC],[test "x$have_tcmalloc" = "xyes"])
