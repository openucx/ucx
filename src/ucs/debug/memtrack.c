/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2013. ALL RIGHTS RESERVED.
* Copyright (C) Huawei Technologies Co., Ltd. 2021.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "memtrack_int.h"

#include <ucs/datastruct/khash.h>
#include <ucs/debug/log.h>
#include <ucs/stats/stats.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/math.h>
#include <ucs/sys/string.h>
#include <ucs/vfs/base/vfs_obj.h>
#ifdef HAVE_MALLOC_H
#include <malloc.h>
#endif
#include <stdio.h>


#define UCS_MEMTRACK_FORMAT_STRING    ("%22s: size: %9lu / %9lu\tcount: %9u / %9u\n")


#define UCS_MEMTRACK_LOG_ZERO_SIZE_ALLOACTION(_size, _ptr, _name) \
    if ((_size) == 0) { \
        ucs_warn("allocated zero-size block %p for %s", _ptr, _name); \
    }


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
    UCS_STATS_NODE_DECLARE(stats)
} ucs_memtrack_context_t;


/* Global context for tracking allocated memory */
static ucs_memtrack_context_t ucs_memtrack_context = {
    .enabled = 0,
    .lock    = PTHREAD_MUTEX_INITIALIZER
};

#ifdef ENABLE_STATS
static ucs_stats_class_t ucs_memtrack_stats_class = {
    .name          = "memtrack",
    .num_counters  = UCS_MEMTRACK_STAT_LAST,
    .class_id      = UCS_STATS_CLASS_ID_INVALID,
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
    ucs_assertv((ret == UCS_KH_PUT_BUCKET_EMPTY) ||
                (ret == UCS_KH_PUT_BUCKET_CLEAR), "ret=%d", ret);
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

static void ucs_memtrack_generate_report()
{
    ucs_status_t status;
    FILE* output_stream;
    const char *next_token;
    int need_close;

    status = ucs_open_output_stream(ucs_global_opts.memtrack_dest,
                                    UCS_LOG_LEVEL_ERROR, &output_stream,
                                    &need_close, &next_token, NULL);
    if (status != UCS_OK) {
        return;
    }

    ucs_memtrack_dump_internal(output_stream);
    if (need_close) {
        fclose(output_stream);
    }
}

static UCS_F_NOINLINE void
ucs_memtrack_do_allocated(void *ptr, size_t size, const char *name)
{
    ucs_memtrack_entry_t *entry;
    khiter_t iter;
    int ret;
    char limit_str[256];

#ifdef UCX_ALLOC_ALIGN
    UCS_STATIC_ASSERT(UCX_ALLOC_ALIGN >= 16);
    UCS_STATIC_ASSERT(ucs_is_pow2_or_zero(UCX_ALLOC_ALIGN));
    ucs_assert(!ucs_check_if_align_pow2((uintptr_t)ptr, UCX_ALLOC_ALIGN));
#endif

    if (ptr == NULL) {
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
    /* do NOT use assert here because it may cause hang due to memtrack malloc
       deadlock */
    if ((ret != UCS_KH_PUT_BUCKET_EMPTY) && (ret != UCS_KH_PUT_BUCKET_CLEAR)) {
        pthread_mutex_unlock(&ucs_memtrack_context.lock);
        ucs_fatal("ret == %d, prev allocation: %s, new allocation: %s", ret,
                  kh_value(&ucs_memtrack_context.ptrs, iter).entry->name, name);
    }

    kh_value(&ucs_memtrack_context.ptrs, iter).entry = entry;
    kh_value(&ucs_memtrack_context.ptrs, iter).size  = size;

    /* update specific and global entries */
    ucs_memtrack_entry_update(entry, size);
    ucs_memtrack_entry_update(&ucs_memtrack_context.total, size);
    if (ucs_memtrack_context.total.size >= ucs_global_opts.memtrack_limit) {
        ucs_memtrack_generate_report();
        ucs_memunits_to_str(ucs_global_opts.memtrack_limit, limit_str,
                            sizeof(limit_str));
        /* disable memtrack to prevent hang */
        ucs_memtrack_context.enabled = 0;
        /* unlock memtrack context to eliminate deadlock */
        pthread_mutex_unlock(&ucs_memtrack_context.lock);
        ucs_fatal("reached memtrack memory limit %s", limit_str);
    }

    UCS_STATS_UPDATE_COUNTER(ucs_memtrack_context.stats, UCS_MEMTRACK_STAT_ALLOCATION_COUNT, 1);
    UCS_STATS_UPDATE_COUNTER(ucs_memtrack_context.stats, UCS_MEMTRACK_STAT_ALLOCATION_SIZE, size);

out_unlock:
    pthread_mutex_unlock(&ucs_memtrack_context.lock);
}


static UCS_F_ALWAYS_INLINE void
ucs_memtrack_allocated_internal(void *ptr, size_t size, const char *name)
{
    if (!ucs_memtrack_is_enabled()) {
        return;
    }

    ucs_memtrack_do_allocated(ptr, size, name);
}

static UCS_F_NOINLINE void ucs_memtrack_do_releasing(void *ptr)
{
    ucs_memtrack_entry_t *entry;
    khiter_t iter;
    size_t size;

    if (ptr == NULL) {
        return;
    }

    pthread_mutex_lock(&ucs_memtrack_context.lock);

    iter = kh_get(ucs_memtrack_ptr_hash, &ucs_memtrack_context.ptrs, (uintptr_t)ptr);
    if (iter == kh_end(&ucs_memtrack_context.ptrs)) {
        /* workaround for coverity - print debug message from unlocked
         * memtrack */
        pthread_mutex_unlock(&ucs_memtrack_context.lock);
        ucs_debug("address %p not found in memtrack ptr hash", ptr);
        return;
    }

    /* remote pointer from hash */
    entry = kh_val(&ucs_memtrack_context.ptrs, iter).entry;
    size  = kh_val(&ucs_memtrack_context.ptrs, iter).size;
    kh_del(ucs_memtrack_ptr_hash, &ucs_memtrack_context.ptrs, iter);

    /* update counts */
    ucs_memtrack_entry_update(entry, -size);
    ucs_memtrack_entry_update(&ucs_memtrack_context.total, -size);

    pthread_mutex_unlock(&ucs_memtrack_context.lock);
}

static UCS_F_ALWAYS_INLINE void ucs_memtrack_releasing_internal(void *ptr)
{
    if (!ucs_memtrack_is_enabled()) {
        return;
    }

    ucs_memtrack_do_releasing(ptr);
}

static void ucs_memtrack_vfs_read(void *obj, ucs_string_buffer_t *strb,
                                  void *arg_ptr, uint64_t arg_u64)
{
    char *buffer;
    size_t size;
    FILE *f;

    f = open_memstream(&buffer, &size);
    ucs_memtrack_dump(f);
    fclose(f);

    ucs_string_buffer_appendf(strb, "%s", buffer);
    free(buffer);
}

static void ucs_memtrack_vfs_init()
{
    ucs_vfs_obj_add_dir(NULL, &ucs_memtrack_context, "ucs/memtrack");
    ucs_vfs_obj_add_ro_file(&ucs_memtrack_context, ucs_memtrack_vfs_read, NULL,
                            0, "all");
}

void *ucs_malloc(size_t size, const char *name)
{
    void *ptr = malloc(size);
    UCS_MEMTRACK_LOG_ZERO_SIZE_ALLOACTION(size, ptr, name);
    ucs_memtrack_allocated_internal(ptr, size, name);
    return ptr;
}

void *ucs_calloc(size_t nmemb, size_t size, const char *name)
{
    void *ptr = calloc(nmemb, size);
    UCS_MEMTRACK_LOG_ZERO_SIZE_ALLOACTION(nmemb * size, ptr, name);
    ucs_memtrack_allocated_internal(ptr, nmemb * size, name);
    return ptr;
}

void *ucs_realloc(void *ptr, size_t size, const char *name)
{
    ucs_memtrack_releasing_internal(ptr);
    ptr = realloc(ptr, size);
    UCS_MEMTRACK_LOG_ZERO_SIZE_ALLOACTION(size, ptr, name);
    ucs_memtrack_allocated_internal(ptr, size, name);
    return ptr;
}

int ucs_posix_memalign(void **ptr, size_t boundary, size_t size, const char *name)
{
    int ret;

#if HAVE_POSIX_MEMALIGN
    ret = posix_memalign(ptr, boundary, size);
#else
#error "Port me"
#endif
    if (ret == 0) {
        ucs_memtrack_allocated_internal(*ptr, size, name);
    }
    return ret;
}

void ucs_free(void *ptr)
{
    ucs_memtrack_releasing_internal(ptr);
    free(ptr);
}

void *ucs_mmap(void *addr, size_t length, int prot, int flags, int fd,
               off_t offset, const char *name)
{
    void *ptr = mmap(addr, length, prot, flags, fd, offset);
    if (ptr != MAP_FAILED) {
        ucs_memtrack_allocated_internal(ptr, length, name);
    }
    return ptr;
}

int ucs_munmap(void *addr, size_t length)
{
    ucs_memtrack_releasing_internal(addr);
    return munmap(addr, length);
}

char *ucs_strdup(const char *src, const char *name)
{
    char *str = strdup(src);
    ucs_memtrack_allocated_internal(str, strlen(str) + 1, name);
    return str;
}

char *ucs_strndup(const char *src, size_t n, const char *name)
{
    char *str = strndup(src, n);
    ucs_memtrack_allocated_internal(str, strlen(str) + 1, name);
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

void ucs_memtrack_dump(FILE* output_stream)
{
    pthread_mutex_lock(&ucs_memtrack_context.lock);
    ucs_memtrack_dump_internal(output_stream);
    pthread_mutex_unlock(&ucs_memtrack_context.lock);
}

void ucs_memtrack_init()
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
                                  ucs_stats_get_root(), "");
    if (status != UCS_OK) {
        return;
    }

    ucs_debug("memtrack enabled");
    ucs_memtrack_context.enabled = 1;

    ucs_memtrack_vfs_init();
}

void ucs_memtrack_cleanup()
{
    ucs_memtrack_entry_t *entry;

    if (!ucs_memtrack_context.enabled) {
        return;
    }

    ucs_vfs_obj_remove(&ucs_memtrack_context);

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
}

int ucs_memtrack_is_enabled()
{
    return ucs_memtrack_context.enabled;
}

int ucs_posix_memalign_realloc(void **ptr, size_t boundary, size_t size,
                               const char *name)
{
    size_t old_size;
    void *tmp;
    int ret;

    /* obtain previous size, to reduce the size for memcpy() below */
    old_size = malloc_usable_size(*ptr);

    /* first try to realloc() - the region may be extended (not guaranteed) */
    tmp = ucs_realloc(*ptr, size, name);
    if (tmp == NULL) {
        return -1;
    }

    /* for some consistency with realloc() - failure leaves a valid pointer */
    *ptr = tmp;

    if (((uintptr_t)tmp % boundary) == 0) {
        return 0;
    }

    ret = ucs_posix_memalign(ptr, boundary, size, name);
    if (ret == 0) {
        ucs_assert(*ptr != NULL);
        memcpy(*ptr, tmp, ucs_min(size, old_size));
        ucs_free(tmp);
    }

    return ret;
}

void ucs_memtrack_allocated(void *ptr, size_t size, const char *name)
{
    ucs_memtrack_allocated_internal(ptr, size, name);
}

void ucs_memtrack_releasing(void *ptr)
{
    ucs_memtrack_releasing_internal(ptr);
}
