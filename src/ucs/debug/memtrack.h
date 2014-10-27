/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2013.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCS_MEMTRACK_H_
#define UCS_MEMTRACK_H_

#include <ucs/sys/sys.h>
#include <ucs/stats/stats.h>
#include <stdio.h>


enum {
    UCS_MEMTRACK_STAT_ALLOCATION_COUNT,
    UCS_MEMTRACK_STAT_ALLOCATION_SIZE,
    UCS_MEMTRACK_STAT_LAST
};


typedef struct ucs_memtrack_entry
{
    char* alloc_name;
    unsigned origin;

    size_t current_size;
    size_t peak_size;

    size_t current_count;
    size_t peak_count;

    struct ucs_memtrack_entry* next;
} ucs_memtrack_entry_t;


typedef struct ucs_memtrack_buffer
{
    size_t magic;  /* Make sure this buffer is "memtracked" */
    size_t length; /* length of user-requested buffer */
    size_t offset; /* Offset between result of memory allocation and the
                      location of this buffer struct (mainly for ucs_memalign) */

    ucs_memtrack_entry_t *entry;
} ucs_memtrack_buffer_t;


#if ENABLE_MEMTRACK

#define UCS_MEMTRACK_NAME(name) , name , __LINE__
#define UCS_MEMTRACK_ARG        , const char* alloc_name, unsigned origin
#define UCS_MEMTRACK_VAL        , alloc_name, origin

#define UCS_MEMTRACK_ADJUST_SIZE_BEFORE(ptr) {\
    if (ucs_memtrack_is_enabled()) *ptr += sizeof(ucs_memtrack_buffer_t);\
    }
#define UCS_MEMTRACK_ADJUST_SIZE_AFTER(ptr)  {\
    if (ucs_memtrack_is_enabled()) *ptr -= sizeof(ucs_memtrack_buffer_t);\
    }

#define ucs_calloc(nmemb, size, name) \
    ucs_memtrack_calloc(nmemb, size UCS_MEMTRACK_NAME(name))
#define ucs_calloc_fwd ucs_memtrack_calloc
#define ucs_malloc(size, name) \
    ucs_memtrack_malloc(size UCS_MEMTRACK_NAME(name))
#define ucs_malloc_cachealigned(size, name) \
    ucs_memtrack_memalign(UCS_SYS_CACHE_LINE_SIZE, size UCS_MEMTRACK_NAME(name))
#define ucs_malloc_fwd ucs_memtrack_malloc
#define ucs_free(ptr) \
    ucs_memtrack_free(ptr)
#define ucs_realloc(ptr, size) \
    ucs_memtrack_realloc(ptr, size)
#define ucs_memalign(boundary, size, name) \
    ucs_memtrack_memalign(boundary, size UCS_MEMTRACK_NAME(name))
#define ucs_memalign_fwd ucs_memtrack_memalign

#define ucs_mmap(addr, length, prot, flags, fd, offset, name) \
    ucs_memtrack_mmap(addr, length, prot, flags, fd, offset UCS_MEMTRACK_NAME(name))
#define ucs_mmap_fwd ucs_memtrack_mmap
#define ucs_mmap64(addr, length, prot, flags, fd, offset, name) \
    ucs_memtrack_mmap(addr, length, prot, flags, fd, offset UCS_MEMTRACK_NAME(name))
#define ucs_munmap(addr, length) \
    ucs_memtrack_munmap(addr, length)

#else

#define UCS_MEMTRACK_NAME(name)
#define UCS_MEMTRACK_ARG
#define UCS_MEMTRACK_VAL
#define UCS_MEMTRACK_ADJUST_SIZE_BEFORE(ptr)
#define UCS_MEMTRACK_ADJUST_SIZE_AFTER(ptr)

#define ucs_calloc(nmemb, size, name) \
    calloc(nmemb, size)
#define ucs_calloc_fwd calloc
#define ucs_malloc(size, name) \
    malloc(size)
#define ucs_malloc_cachealigned(size, name) \
    memalign(UCS_SYS_CACHE_LINE_SIZE, size)
#define ucs_malloc_fwd malloc
#define ucs_free(ptr) \
    free(ptr)
#define ucs_realloc(ptr, size) \
    realloc(ptr, size)
#define ucs_memalign(boundary, size, name) \
    memalign(boundary, size)
#define ucs_memalign_fwd memalign

#define ucs_mmap(addr, length, prot, flags, fd, offset, name) \
    mmap(addr, length, prot, flags, fd, offset UCS_MEMTRACK_NAME(name))
#define ucs_mmap_fwd mmap
#define ucs_mmap64(addr, length, prot, flags, fd, offset, name) \
    mmap(addr, length, prot, flags, fd, offset UCS_MEMTRACK_NAME(name))
#define ucs_munmap(addr, length) \
    munmap(addr, length)

#endif /* ENABLE_MEMTRACK */

#define UCS_MEMTRACK_ADJUST_PTR_BEFORE(ptr) UCS_MEMTRACK_ADJUST_SIZE_AFTER(ptr)
#define UCS_MEMTRACK_ADJUST_PTR_AFTER(ptr) UCS_MEMTRACK_ADJUST_SIZE_BEFORE(ptr)

/**
 * Start trakcing memory (or increment reference count).
 */
void ucs_memtrack_init();

/**
 * Stop trakcing memory (or decrement reference count).
 */
void ucs_memtrack_cleanup();

/*
 * Check if memtrack is enbled at the moment.
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
 * Memory allocation registration for tracking.
 *
 * @param ptr            Pointer to note as allocated.
 * @param size           Size of the requested allocation.
 * @param alloc_name    Name for this allocation command.
 *
 */
void ucs_memtrack_record_alloc(ucs_memtrack_buffer_t* buffer, size_t size
                                 UCS_MEMTRACK_ARG);

/**
 * Memory deallocation registration for tracking.
 *
 * @param ptr            Pointer to note as deallocated.
 */
ucs_memtrack_entry_t* ucs_memtrack_record_dealloc(ucs_memtrack_buffer_t* buffer);

/**
 * Clear memory allocation request.
 *
 * @param nmemb          Amount of allocated slots requested.
 * @param size           Size of allocation slots requested.
 * @param alloc_name    Name for this allocation command.
 * @param origin         Origin of the allocation command (used for id).
 */
void *ucs_memtrack_calloc(size_t nmemb, size_t size UCS_MEMTRACK_ARG);

/**
 * Memory allocation request.
 *
 * @param size           Size of allocation requested.
 * @param alloc_name    Name for this allocation command.
 * @param origin         Origin of the allocation command (used for id).
 */
void *ucs_memtrack_malloc(size_t size UCS_MEMTRACK_ARG);

/**
 * Memory deallocation request.
 *
 * @param ptr            Pointer for note as allocated.
 */
void ucs_memtrack_free(void *ptr);

/**
 * Memory reallocation request.
 *
 * @param ptr            Pointer to reallocate.
 * @param size           Size of the requested allocation.
 */
void *ucs_memtrack_realloc(void *ptr, size_t size);

/**
 * Memory alignment request.
 *
 * @param boundary       Size to align memory to.
 * @param size           Size of the requested allocation.
 * @param alloc_name    Name for this allocation command.
 * @param origin         Origin of the allocation command (used for id).
 */
void *ucs_memtrack_memalign(size_t boundary, size_t size UCS_MEMTRACK_ARG);

/**
 * Memory mapping request.
 *
 * @param addr           Address hint for mapping.
 * @param length         Size of the requested allocation.
 * @param prot           Memory protection mask.
 * @param flags          Other flags.
 * @param fd             File to map.
 * @param offset         Offset within the file.
 * @param alloc_name    Name for this allocation command.
 * @param origin         Origin of the allocation command (used for id).
 */
void *ucs_memtrack_mmap(void *addr, size_t length, int prot, int flags,
                        int fd, off_t offset UCS_MEMTRACK_ARG);

/**
 * Memory mapping request (for 64 bit).
 *
 * @param addr           Address hint for mapping.
 * @param length         Size of the requested allocation.
 * @param prot           Memory protection mask.
 * @param flags          Other flags.
 * @param fd             File to map.
 * @param offset         Offset within the file.
 * @param alloc_name    Name for this allocation command.
 * @param origin         Origin of the allocation command (used for id).
 */
#ifdef __USE_LARGEFILE64
void *ucs_memtrack_mmap64(void *addr, size_t length, int prot, int flags,
                          int fd, uint64_t offset UCS_MEMTRACK_ARG);
#endif

/**
 * Memory unmaping request.
 *
 * @param addr           Address to unmap.
 * @param length         Size of the requested allocation.
 */
int ucs_memtrack_munmap(void *addr, size_t length);

#endif
