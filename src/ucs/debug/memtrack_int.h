/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef UCS_MEMTRACK_INT_H_
#define UCS_MEMTRACK_INT_H_

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/sys/compiler_def.h>
#include <ucs/debug/memtrack.h>
#include <stdlib.h>
#include <stdio.h>


BEGIN_C_DECLS

/** @file memtrack_int.h */

enum {
    UCS_MEMTRACK_STAT_ALLOCATION_COUNT,
    UCS_MEMTRACK_STAT_ALLOCATION_SIZE,
    UCS_MEMTRACK_STAT_LAST
};


/**
 * Allocation site entry
 */
typedef struct ucs_memtrack_entry {
    size_t                  size;       /* currently allocated total size */
    size_t                  peak_size;  /* peak allocated total size */
    unsigned                count;      /* number of currently allocated blocks */
    unsigned                peak_count; /* peak number of allocated blocks */
    char                    name[0];    /* allocation name */
} ucs_memtrack_entry_t;


/**
 * Start tracking memory (or increment reference count).
 */
void ucs_memtrack_init();


/**
 * Stop tracking memory (or decrement reference count).
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


/*
 * Memory allocation replacements. Their interface is the same as the originals,
 * except the additional parameter which specifies the allocation name.
 */
void *ucs_malloc(size_t size, const char *name);
void *ucs_calloc(size_t nmemb, size_t size, const char *name);
void *ucs_realloc(void *ptr, size_t size, const char *name);
int ucs_posix_memalign(void **ptr, size_t boundary, size_t size,
                       const char *name);
void ucs_free(void *ptr);
void *ucs_mmap(void *addr, size_t length, int prot, int flags, int fd,
               off_t offset, const char *name);
int ucs_munmap(void *addr, size_t length);
char *ucs_strdup(const char *src, const char *name);
char *ucs_strndup(const char *src, size_t n, const char *name);

/*
 * The functions below have no native implementation, they apply to both cases.
 */
int ucs_posix_memalign_realloc(void **ptr, size_t boundary, size_t size,
                               const char *name);

END_C_DECLS

#endif /* UCS_MEMTRACK_INT_H_ */
