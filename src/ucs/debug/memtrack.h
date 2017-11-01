/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2013.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef UCS_MEMTRACK_H_
#define UCS_MEMTRACK_H_

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/sys/compiler_def.h>

#include <sys/types.h>
#include <malloc.h>
#include <stdlib.h>
#include <stdio.h>

BEGIN_C_DECLS

enum {
    UCS_MEMTRACK_STAT_ALLOCATION_COUNT,
    UCS_MEMTRACK_STAT_ALLOCATION_SIZE,
    UCS_MEMTRACK_STAT_LAST
};

#define UCS_MEMTRACK_NAME_MAX  31

/**
 * Allocation site entry.
 */
typedef struct ucs_memtrack_entry ucs_memtrack_entry_t;
struct ucs_memtrack_entry {
    char                  name[UCS_MEMTRACK_NAME_MAX];
    size_t                size;
    size_t                peak_size;
    size_t                count;
    size_t                peak_count;
    ucs_memtrack_entry_t  *next;
};


/**
 * Initialize the total allocations structure.
 */
void ucs_memtrack_total_reset(ucs_memtrack_entry_t* total);


#if ENABLE_MEMTRACK

#define UCS_MEMTRACK_ARG        , const char* alloc_name
#define UCS_MEMTRACK_VAL        , alloc_name
#define UCS_MEMTRACK_VAL_ALWAYS alloc_name
#define UCS_MEMTRACK_NAME(_n)   , _n


/**
 * Start trakcing memory (or increment reference count).
 */
void ucs_memtrack_init();

/**
 * Stop trakcing memory (or decrement reference count).
 */
void ucs_memtrack_cleanup();

/*
 * Check if memtrack is enabled at the moment.
 */
int ucs_memtrack_is_enabled();

/**
 * Print a summary of memory tracked so far.
 *
 * @param output         Stream to direct output to.
 */
void ucs_memtrack_dump(FILE* output);

/**
 * Calculates the total of buffers currently tracked.
 *
 * @param total          Entry (pre-allocated) to place results in.
 */
void ucs_memtrack_total(ucs_memtrack_entry_t* total);

/**
 * Adjust size before doing custom allocation. Need to be called in order to
 * obtain the size of a custom allocation to have room for memtrack descriptor.
 */
size_t ucs_memtrack_adjust_alloc_size(size_t size);

/**
 * Track custom allocation. Need to be called after custom allocation returns,
 * it will adjust the pointer and size to user buffer instead of the memtrack
 * descriptor.
 */
void ucs_memtrack_allocated(void **ptr_p, size_t *size_p, const char *name);

/**
 * Track release of custom allocation. Need to be called before actually
 * releasing the memory.
 */
void ucs_memtrack_releasing(void **ptr_p);


/**
 * Track release of custom allocation. Need to be called before actually
 * releasing the memory. Unlike @ref ucs_memtrack_releasing(), the pointer passed
 * to this function is the actual memory block including memtrack header.
 */
void ucs_memtrack_releasing_adjusted(void *ptr);


/*
 * Memory allocation replacements. Their interface is the same as the originals,
 * except the additional parameter which specifies the allocation name.
 */
void *ucs_malloc(size_t size, const char *name);
void *ucs_calloc(size_t nmemb, size_t size, const char *name);
void *ucs_realloc(void *ptr, size_t size, const char *name);
void *ucs_memalign(size_t boundary, size_t size, const char *name);
void ucs_free(void *ptr);
void *ucs_mmap(void *addr, size_t length, int prot, int flags, int fd,
               off_t offset, const char *name);
#ifdef __USE_LARGEFILE64
void *ucs_mmap64(void *addr, size_t size, int prot, int flags, int fd,
                 off64_t offset, const char *name);
#endif
int ucs_munmap(void *addr, size_t length);
char *ucs_strdup(const char *src, const char *name);

#else

#define UCS_MEMTRACK_ARG
#define UCS_MEMTRACK_VAL
#define UCS_MEMTRACK_VAL_ALWAYS                    ""
#define UCS_MEMTRACK_NAME(_n)

#define ucs_memtrack_init()                        UCS_EMPTY_STATEMENT
#define ucs_memtrack_cleanup()                     UCS_EMPTY_STATEMENT
#define ucs_memtrack_is_enabled()                  0
#define ucs_memtrack_dump(_output)                 UCS_EMPTY_STATEMENT
#define ucs_memtrack_total(_total)                 ucs_memtrack_total_init(_total)

#define ucs_memtrack_adjust_alloc_size(_size)      (_size)
#define ucs_memtrack_allocated(_ptr_p, _sz_p, ...) UCS_EMPTY_STATEMENT
#define ucs_memtrack_releasing(_ptr)               UCS_EMPTY_STATEMENT
#define ucs_memtrack_releasing_adjusted(_ptr)      UCS_EMPTY_STATEMENT

#define ucs_malloc(_s, ...)                        malloc(_s)
#define ucs_calloc(_n, _s, ...)                    calloc(_n, _s)
#define ucs_realloc(_p, _s, ...)                   realloc(_p, _s)
#define ucs_memalign(_b, _s, ...)                  memalign(_b, _s)
#define ucs_free(_p)                               free(_p)
#define ucs_mmap(_a, _l, _p, _fl, _fd, _o, ...)    mmap(_a, _l, _p, _fl, _fd, _o)
#define ucs_mmap64(_a, _l, _p, _fl, _fd, _o, ...)  mmap64(_a, _l, _p, _fl, _fd, _o)
#define ucs_munmap(_a, _l)                         munmap(_a, _l)
#define ucs_strdup(_src, ...)                      strdup(_src)

#endif /* ENABLE_MEMTRACK */

END_C_DECLS

#endif
