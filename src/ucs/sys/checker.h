/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_CHECKER_H_
#define UCS_CHECKER_H_

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

/*
 * Valgrind support
 */
#ifndef NVALGRIND
#  include <valgrind/memcheck.h>
#  ifndef VALGRIND_MAKE_MEM_DEFINED
#    define VALGRIND_MAKE_MEM_DEFINED(p, n)   VALGRIND_MAKE_READABLE(p, n)
#  endif
#  ifndef VALGRIND_MAKE_MEM_UNDEFINED
#    define VALGRIND_MAKE_MEM_UNDEFINED(p, n) VALGRIND_MAKE_WRITABLE(p, n)
#  endif
#else
#  define VALGRIND_MAKE_MEM_DEFINED(p, n)
#  define VALGRIND_MAKE_MEM_UNDEFINED(p, n)
#  define VALGRIND_MAKE_MEM_NOACCESS(p, n)
#  define VALGRIND_CREATE_MEMPOOL(n,p,x)
#  define VALGRIND_DESTROY_MEMPOOL(p)
#  define VALGRIND_MEMPOOL_ALLOC(n,p,x)
#  define VALGRIND_MEMPOOL_FREE(n,p)
#  define VALGRIND_MALLOCLIKE_BLOCK(p,s,r,z)
#  define VALGRIND_FREELIKE_BLOCK(p,r)
#  define VALGRIND_CHECK_MEM_IS_DEFINED(p, n) ({(uintptr_t)0;})
#  define VALGRIND_COUNT_ERRORS              0
#  define VALGRIND_COUNT_LEAKS(a,b,c,d)      { a = b = c = d = 0; }
#  define RUNNING_ON_VALGRIND                0
#  define VALGRIND_PRINTF(...)
#endif


/*
 * BullsEye Code Coverage tool
 */
#if _BullseyeCoverage
#define BULLSEYE_ON                          1
#define BULLSEYE_EXCLUDE_START               #pragma BullseyeCoverage off
#define BULLSEYE_EXCLUDE_END                 #pragma BullseyeCoverage on
#define BULLSEYE_EXCLUDE_BLOCK_START         "BullseyeCoverage save off";
#define BULLSEYE_EXCLUDE_BLOCK_END           "BullseyeCoverage restore";
#else
#define BULLSEYE_ON                          0
#define BULLSEYE_EXCLUDE_START
#define BULLSEYE_EXCLUDE_END
#define BULLSEYE_EXCLUDE_BLOCK_START
#define BULLSEYE_EXCLUDE_BLOCK_END
#endif


#endif
