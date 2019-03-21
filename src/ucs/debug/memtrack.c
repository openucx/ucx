/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2013.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "memtrack.h"

#include <ucs/datastruct/khash.h>
#include <ucs/debug/log.h>
#include <ucs/stats/stats.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/math.h>
#include <malloc.h>
#include <stdio.h>


#if ENABLE_MEMTRACK

#define UCS_MEMTRACK_FORMAT_STRING    ("%22s: size: %9lu / %9lu\tcount: %9u / %9u\n")


typedef struct ucs_memtrack_ptr {
    size_t                  size;   /* Length of allocated buffer */
    ucs_memtrack_entry_t    *entry; /* Entry which tracks this allocation */
} ucs_memtrack_ptr_t;

KHASH_MAP_INIT_INT64(ucs_memtrack_ptr_hash, ucs_memtrack_ptr_t)
KHASH_MAP_INIT_STR(ucs_memtrack_entry_hash, ucs_memtrack_entry_t*);

typedef struct ucs_memtrack_context {
    int                              enabled;
    pthread_mutex_t                  lock;
    ucs_memtrack_entry_t             total;
    khash_t(ucs_memtrack_ptr_hash)   ptrs;
    khash_t(ucs_memtrack_entry_hash) entries;
    UCS_STATS_NODE_DECLARE(stats);
} ucs_memtrack_context_t;


/* Global context for tracking allocated memory */
static ucs_memtrack_context_t ucs_memtrack_context = {
    .enabled = 0,
    .lock    = PTHREAD_MUTEX_INITIALIZER,
    .total   = {0}
};

#if ENABLE_STATS
static ucs_stats_class_t ucs_memtrack_stats_class = {
    .name = "memtrack",
    .num_counters = UCS_MEMTRACK_STAT_LAST,
    .counter_names = {
        [UCS_MEMTRACK_STAT_ALLOCATION_COUNT] = "alloc_cnt",
        [UCS_MEMTRACK_STAT_ALLOCATION_SIZE]  = "alloc_size"
    }
};
#endif

static void ucs_memtrack_entry_reset(ucs_memtrack_entry_t *entry)
{
    entry->size       = 0;
    entry->peak_size  = 0;
    entry->count      = 0;
    entry->peak_count = 0;
}

static ucs_memtrack_entry_t* ucs_memtrack_entry_get(const char* name)
{
    ucs_memtrack_entry_t *entry;
    khiter_t iter;
    int ret;

    iter = kh_get(ucs_memtrack_entry_hash, &ucs_memtrack_context.entries, name);
    if (iter != kh_end(&ucs_memtrack_context.entries)) {
        return kh_val(&ucs_memtrack_context.entries, iter);
    }

    entry = malloc(sizeof(*entry) + strlen(name) + 1);
    if (entry == NULL) {
        return NULL;
    }

    ucs_memtrack_entry_reset(entry);
    strcpy(entry->name, name);

    iter = kh_put(ucs_memtrack_entry_hash, &ucs_memtrack_context.entries,
                  entry->name, &ret);
    ucs_assertv(ret == 1 || ret == 2, "ret=%d", ret);
    kh_val(&ucs_memtrack_context.entries, iter) = entry;

    return entry;
}

static void ucs_memtrack_entry_update(ucs_memtrack_entry_t *entry, ssize_t size)
{
    int count = (size < 0) ? -1 : 1;

    ucs_assert((int)entry->count    >= -count);
    ucs_assert((ssize_t)entry->size >= -size);
    entry->count      += count;
    entry->size       += size;
    entry->peak_count  = ucs_max(entry->peak_count, entry->count);
    entry->peak_size   = ucs_max(entry->peak_size,  entry->size);
}

void ucs_memtrack_allocated(void *ptr, size_t size, const char *name)
{
    ucs_memtrack_entry_t *entry;
    khiter_t iter;
    int ret;

    if ((ptr == NULL) || !ucs_memtrack_is_enabled()) {
        return;
    }

    pthread_mutex_lock(&ucs_memtrack_context.lock);

    entry = ucs_memtrack_entry_get(name);
    if (entry == NULL) {
        goto out_unlock;
    }

    /* Add pointer to hash */
    iter = kh_put(ucs_memtrack_ptr_hash, &ucs_memtrack_context.ptrs,
                  (uintptr_t)ptr, &ret);
    ucs_assertv(ret == 1 || ret == 2, "ret=%d", ret);
    kh_value(&ucs_memtrack_context.ptrs, iter).entry = entry;
    kh_value(&ucs_memtrack_context.ptrs, iter).size  = size;

    /* update specific and global entries */
    ucs_memtrack_entry_update(entry, size);
    ucs_memtrack_entry_update(&ucs_memtrack_context.total, size);

    UCS_STATS_UPDATE_COUNTER(ucs_memtrack_context.stats, UCS_MEMTRACK_STAT_ALLOCATION_COUNT, 1);
    UCS_STATS_UPDATE_COUNTER(ucs_memtrack_context.stats, UCS_MEMTRACK_STAT_ALLOCATION_SIZE, size);

out_unlock:
    pthread_mutex_unlock(&ucs_memtrack_context.lock);
}

void ucs_memtrack_releasing(void* ptr)
{
    ucs_memtrack_entry_t *entry;
    khiter_t iter;
    size_t size;

    if ((ptr == NULL) || !ucs_memtrack_is_enabled()) {
        return;
    }

    pthread_mutex_lock(&ucs_memtrack_context.lock);

    iter = kh_get(ucs_memtrack_ptr_hash, &ucs_memtrack_context.ptrs, (uintptr_t)ptr);
    if (iter == kh_end(&ucs_memtrack_context.ptrs)) {
        ucs_debug("address %p not found in memtrack ptr hash", ptr);
        goto out_unlock;
    }

    /* remote pointer from hash */
    entry = kh_val(&ucs_memtrack_context.ptrs, iter).entry;
    size  = kh_val(&ucs_memtrack_context.ptrs, iter).size;
    kh_del(ucs_memtrack_ptr_hash, &ucs_memtrack_context.ptrs, iter);

    /* update counts */
    ucs_memtrack_entry_update(entry, -size);
    ucs_memtrack_entry_update(&ucs_memtrack_context.total, -size);

out_unlock:
    pthread_mutex_unlock(&ucs_memtrack_context.lock);
}

void *ucs_malloc(size_t size, const char *name)
{
    void *ptr = malloc(size);
    ucs_memtrack_allocated(ptr, size, name);
    return ptr;
}

void *ucs_calloc(size_t nmemb, size_t size, const char *name)
{
    void *ptr = calloc(nmemb, size);
    ucs_memtrack_allocated(ptr, nmemb * size, name);
    return ptr;
}

void *ucs_realloc(void *ptr, size_t size, const char *name)
{
    ucs_memtrack_releasing(ptr);
    ptr = realloc(ptr, size);
    ucs_memtrack_allocated(ptr, size, name);
    return ptr;
}

void *ucs_memalign(size_t boundary, size_t size, const char *name)
{
    void *ptr = memalign(boundary, size);
    ucs_memtrack_allocated(ptr, size, name);
    return ptr;
}

void ucs_free(void *ptr)
{
    ucs_memtrack_releasing(ptr);
    free(ptr);
}

void *ucs_mmap(void *addr, size_t length, int prot, int flags, int fd,
               off_t offset, const char *name)
{
    void *ptr = mmap(addr, length, prot, flags, fd, offset);
    if (ptr != MAP_FAILED) {
        ucs_memtrack_allocated(ptr, length, name);
    }
    return ptr;
}

int ucs_munmap(void *addr, size_t length)
{
    ucs_memtrack_releasing(addr);
    return munmap(addr, length);
}

char *ucs_strdup(const char *src, const char *name)
{
    char *str = strdup(src);
    ucs_memtrack_allocated(str, strlen(str) + 1, name);
    return str;
}

char *ucs_strndup(const char *src, size_t n, const char *name)
{
    char *str = strndup(src, n);
    ucs_memtrack_allocated(str, strlen(str) + 1, name);
    return str;
}

void ucs_memtrack_total(ucs_memtrack_entry_t* total)
{
    if (!ucs_memtrack_is_enabled()) {
        return;
    }

    pthread_mutex_lock(&ucs_memtrack_context.lock);
    *total = ucs_memtrack_context.total;
    pthread_mutex_unlock(&ucs_memtrack_context.lock);
}

static int ucs_memtrack_cmp_entries(const void *ptr1, const void *ptr2)
{
    ucs_memtrack_entry_t * const *e1 = ptr1;
    ucs_memtrack_entry_t * const *e2 = ptr2;

    return (int)((ssize_t)(*e2)->peak_size - (ssize_t)(*e1)->peak_size);
}

static void ucs_memtrack_dump_internal(FILE* output_stream)
{
    ucs_memtrack_entry_t *entry, **all_entries;
    unsigned num_entries, i;

    if (!ucs_memtrack_is_enabled()) {
        return;
    }

    /* collect all entries to one array */
    all_entries = ucs_alloca(sizeof(*all_entries) *
                             kh_size(&ucs_memtrack_context.entries));
    num_entries = 0;
    kh_foreach_value(&ucs_memtrack_context.entries, entry, {
        all_entries[num_entries++] = entry;
    });
    ucs_assert(num_entries <= kh_size(&ucs_memtrack_context.entries));

    /* sort entries according to peak size */
    qsort(all_entries, num_entries, sizeof(*all_entries), ucs_memtrack_cmp_entries);

    /* print title */
    fprintf(output_stream, "%31s current / peak  %16s current / peak\n", "", "");
    fprintf(output_stream, UCS_MEMTRACK_FORMAT_STRING, "TOTAL",
            ucs_memtrack_context.total.size, ucs_memtrack_context.total.peak_size,
            ucs_memtrack_context.total.count, ucs_memtrack_context.total.peak_count);

    /* print sorted entries */
    for (i = 0; i < num_entries; ++i) {
        entry = all_entries[i];
        fprintf(output_stream, UCS_MEMTRACK_FORMAT_STRING, entry->name,
                entry->size, entry->peak_size, entry->count, entry->peak_count);
    }
}

void ucs_memtrack_dump(FILE* output_stream)
{
    pthread_mutex_lock(&ucs_memtrack_context.lock);
    ucs_memtrack_dump_internal(output_stream);
    pthread_mutex_unlock(&ucs_memtrack_context.lock);
}

static void ucs_memtrack_generate_report(void)
{
    ucs_status_t status;
    FILE* output_stream;
    const char *next_token;
    int need_close;

    status = ucs_open_output_stream(ucs_global_opts.memtrack_dest,
                                    UCS_LOG_LEVEL_ERROR, &output_stream,
                                    &need_close, &next_token);
    if (status != UCS_OK) {
        return;
    }

    ucs_memtrack_dump_internal(output_stream);
    if (need_close) {
        fclose(output_stream);
    }
}

void ucs_memtrack_init(void)
{
    ucs_status_t status;

    ucs_assert(ucs_memtrack_context.enabled == 0);

    if (!strcmp(ucs_global_opts.memtrack_dest, "")) {
        ucs_trace("memtrack disabled");
        ucs_memtrack_context.enabled = 0;
        return;
    }

    // TODO use ucs_memtrack_entry_reset
    ucs_memtrack_entry_reset(&ucs_memtrack_context.total);
    kh_init_inplace(ucs_memtrack_ptr_hash, &ucs_memtrack_context.ptrs);
    kh_init_inplace(ucs_memtrack_entry_hash, &ucs_memtrack_context.entries);

    status = UCS_STATS_NODE_ALLOC(&ucs_memtrack_context.stats,
                                  &ucs_memtrack_stats_class,
                                  ucs_stats_get_root());
    if (status != UCS_OK) {
        return;
    }

    ucs_debug("memtrack enabled");
    ucs_memtrack_context.enabled = 1;
}

void ucs_memtrack_cleanup(void)
{
    ucs_memtrack_entry_t *entry;

    if (!ucs_memtrack_context.enabled) {
        return;
    }

    pthread_mutex_lock(&ucs_memtrack_context.lock);

    ucs_memtrack_generate_report();

    /* disable before releasing the stats node */
    ucs_memtrack_context.enabled = 0;
    UCS_STATS_NODE_FREE(ucs_memtrack_context.stats);

    /* cleanup entries */
    kh_foreach_value(&ucs_memtrack_context.entries, entry, {
         free(entry);
    });

    /* destroy hash tables */
    kh_destroy_inplace(ucs_memtrack_entry_hash, &ucs_memtrack_context.entries);
    kh_destroy_inplace(ucs_memtrack_ptr_hash, &ucs_memtrack_context.ptrs);

    pthread_mutex_unlock(&ucs_memtrack_context.lock);
}

int ucs_memtrack_is_enabled(void)
{
    return ucs_memtrack_context.enabled;
}

#endif
