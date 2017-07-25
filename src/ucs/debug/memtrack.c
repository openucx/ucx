/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2013.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "memtrack.h"

#include <stdio.h>
#include <string.h>
#include <malloc.h>

#include <ucs/debug/log.h>
#include <ucs/stats/stats.h>
#include <ucs/datastruct/list.h>
#include <ucs/datastruct/mpool.h>
#include <ucs/datastruct/sglib_wrapper.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/math.h>
#include <ucs/sys/checker.h>
#include <ucs/sys/string.h>
#include <ucs/sys/sys.h>


#if ENABLE_MEMTRACK

#define UCS_MEMTRACK_MAGIC            0x1ee7beefa880feedULL
#define UCS_MEMTRACK_FORMAT_STRING    ("%22s: size: %9lu / %9lu\tcount: %9lu / %9lu\n")
#define UCS_MEMTRACK_ENTRY_HASH_SIZE  127


typedef struct ucs_memtrack_buffer {
    uint64_t              magic;  /* Make sure this buffer is "memtracked" */
    size_t                size; /* length of user-requested buffer */
    off_t                 offset; /* Offset between result of memory allocation and the
                                     location of this buffer struct (mainly for ucs_memalign) */
    ucs_memtrack_entry_t  *entry; /* Entry which tracks this allocation */
} ucs_memtrack_buffer_t;


typedef struct ucs_memtrack_context {
    int                     enabled;
    pthread_mutex_t         lock;
    ucs_memtrack_entry_t    *entries[UCS_MEMTRACK_ENTRY_HASH_SIZE];
    UCS_STATS_NODE_DECLARE(stats);
} ucs_memtrack_context_t;


/* Global context for tracking allocated memory */
static ucs_memtrack_context_t ucs_memtrack_context = {
    .enabled = 0,
    .lock    = PTHREAD_MUTEX_INITIALIZER
};

SGLIB_DEFINE_LIST_PROTOTYPES(ucs_memtrack_entry_t, ucs_memtrack_entry_compare, next)
SGLIB_DEFINE_HASHED_CONTAINER_PROTOTYPES(ucs_memtrack_entry_t,
                                         UCS_MEMTRACK_ENTRY_HASH_SIZE,
                                         ucs_memtrack_entry_hash)

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


static inline ucs_memtrack_entry_t* ucs_memtrack_entry_new(const char* name)
{
    ucs_memtrack_entry_t *entry;

    entry = malloc(sizeof(*entry));
    if (entry == NULL) {
        return NULL;
    }

    entry->size       = 0;
    entry->peak_size  = 0;
    entry->count      = 0;
    entry->peak_count = 0;
    ucs_snprintf_zero(entry->name, UCS_MEMTRACK_NAME_MAX, "%s", name);
    sglib_hashed_ucs_memtrack_entry_t_add(ucs_memtrack_context.entries, entry);
    return entry;
}

static void ucs_memtrack_record_alloc(ucs_memtrack_buffer_t* buffer, size_t size,
                                      off_t offset, const char *name)
{
    ucs_memtrack_entry_t *entry, search;
    if (!ucs_memtrack_is_enabled()) {
        goto out;
    }

    if (strlen(name) >= UCS_MEMTRACK_NAME_MAX - 1) {
        ucs_fatal("memory allocation name too long: '%s' (len: %ld, max: %d)",
                  name, strlen(name), UCS_MEMTRACK_NAME_MAX - 1);
    }

    ucs_assert(buffer != NULL);
    ucs_assert(ucs_memtrack_context.entries != NULL); // context initialized
    pthread_mutex_lock(&ucs_memtrack_context.lock);

    ucs_snprintf_zero(search.name, UCS_MEMTRACK_NAME_MAX, "%s", name);
    entry = sglib_hashed_ucs_memtrack_entry_t_find_member(ucs_memtrack_context.entries,
                                                          &search);
    if (entry == NULL) {
        entry = ucs_memtrack_entry_new(name);
        if (entry == NULL) {
            goto out_unlock;
        }
    }

    ucs_assert(!strcmp(name, entry->name));
    buffer->magic   = UCS_MEMTRACK_MAGIC;
    buffer->size    = size;
    buffer->offset  = offset;
    buffer->entry   = entry;
    VALGRIND_MAKE_MEM_NOACCESS(buffer, sizeof(*buffer));

    /* Update total count */
    entry->count++;
    UCS_STATS_UPDATE_COUNTER(ucs_memtrack_context.stats, UCS_MEMTRACK_STAT_ALLOCATION_COUNT, 1);
    entry->peak_count = ucs_max(entry->peak_count, entry->count);

    /* Update total size */
    entry->size += size;
    UCS_STATS_UPDATE_COUNTER(ucs_memtrack_context.stats, UCS_MEMTRACK_STAT_ALLOCATION_SIZE, size);
    entry->peak_size = ucs_max(entry->peak_size, entry->size);

out_unlock:
    pthread_mutex_unlock(&ucs_memtrack_context.lock);
out:
    UCS_EMPTY_STATEMENT;
}

static ucs_memtrack_entry_t*
ucs_memtrack_record_release(ucs_memtrack_buffer_t *buffer, size_t size)
{
    ucs_memtrack_entry_t *entry;

    if (!ucs_memtrack_is_enabled()) {
        return NULL;
    }

    pthread_mutex_lock(&ucs_memtrack_context.lock);
    VALGRIND_MAKE_MEM_DEFINED(buffer, sizeof(*buffer));

    ucs_assert_always(buffer->magic == UCS_MEMTRACK_MAGIC);
    buffer->magic = UCS_MEMTRACK_MAGIC + 1; /* protect from double free */
    if (size != 0) {
        ucs_assert(buffer->size == size);
    }

    entry = buffer->entry;

    /* Update total count */
    ucs_assert(entry->count >= 1);
    --entry->count;

    /* Update total size */
    ucs_assert(entry->size >= buffer->size);
    entry->size -= buffer->size;

    pthread_mutex_unlock(&ucs_memtrack_context.lock);
    return entry;
}

void *ucs_malloc(size_t size, const char *name)
{
    ucs_memtrack_buffer_t *buffer;

    buffer = malloc(size + (ucs_memtrack_is_enabled() ? sizeof(*buffer) : 0));
    if ((buffer == NULL) || (!ucs_memtrack_is_enabled())) {
        return buffer;
    }

    ucs_memtrack_record_alloc(buffer, size, 0, name);
    return buffer + 1;
}

void *ucs_calloc(size_t nmemb, size_t size, const char *name)
{
    ucs_memtrack_buffer_t *buffer;

    buffer = calloc(1, nmemb * size + (ucs_memtrack_is_enabled() ? sizeof(*buffer) : 0));
    if ((buffer == NULL) || (!ucs_memtrack_is_enabled())) {
        return buffer;
    }

    ucs_memtrack_record_alloc(buffer, nmemb * size, 0, name);
    return buffer + 1;
}

void *ucs_realloc(void *ptr, size_t size, const char *name)
{
    ucs_memtrack_buffer_t *buffer = (ucs_memtrack_buffer_t*)ptr - 1;
    ucs_memtrack_entry_t *entry;

    if (!ucs_memtrack_is_enabled()) {
        return realloc(ptr, size);
    }

    if (ptr == NULL) {
        return ucs_malloc(size, name);
    }

    entry = ucs_memtrack_record_release(buffer, 0);

    buffer = realloc((void*)buffer - buffer->offset, size + sizeof(*buffer));
    if (buffer == NULL) {
        return NULL;
    }

    ucs_memtrack_record_alloc(buffer, size, 0, entry->name);
    return buffer + 1;
}

void *ucs_memalign(size_t boundary, size_t size, const char *name)
{
    ucs_memtrack_buffer_t *buffer;
    off_t offset;

    if (!ucs_memtrack_is_enabled()) {
        return memalign(boundary, size);
    }

    if (boundary > sizeof(*buffer)) {
        buffer = memalign(boundary, size + boundary);
        offset = boundary - sizeof(*buffer);
    } else {
        if (sizeof(*buffer) % boundary != 0) {
            offset = boundary - (sizeof(*buffer) % boundary);
        } else {
            offset = 0;
        }
        buffer = memalign(boundary, size + sizeof(*buffer) + offset);
    }
    if ((buffer == NULL) || (!ucs_memtrack_is_enabled())) {
        return buffer;
    }

    buffer = (void*)buffer + offset;
    ucs_memtrack_record_alloc(buffer, size, offset, name);
    return buffer + 1;
}

void ucs_free(void *ptr)
{
    ucs_memtrack_buffer_t *buffer;

    if ((ptr == NULL) || !ucs_memtrack_is_enabled()) {
        free(ptr);
        return;
    }

    buffer = (ucs_memtrack_buffer_t*)ptr - 1;
    ucs_memtrack_record_release(buffer, 0);
    free((void*)buffer - buffer->offset);
}

void *ucs_mmap(void *addr, size_t length, int prot, int flags, int fd,
               off_t offset, const char *name)
{
    ucs_memtrack_buffer_t *buffer;

    if (ucs_memtrack_is_enabled() &&
        ((flags & MAP_FIXED) || !(prot & PROT_WRITE))) {
        return MAP_FAILED;
    }

    buffer = mmap(addr, length + (ucs_memtrack_is_enabled() ? sizeof(*buffer) : 0),
               prot, flags, fd, offset);
    if ((buffer == MAP_FAILED) || (!ucs_memtrack_is_enabled())) {
        return buffer;
    }

    if (fd > 0) {
        memmove(buffer + 1, buffer, length);
    }

    ucs_memtrack_record_alloc(buffer, length, 0, name);
    return buffer + 1;
}

#ifdef __USE_LARGEFILE64
void *ucs_mmap64(void *addr, size_t size, int prot, int flags, int fd,
                 off64_t offset, const char *name)
{
    ucs_memtrack_buffer_t *buffer;

    if ((flags & MAP_FIXED) || !(prot & PROT_WRITE)) {
        return NULL;
    }

    buffer = mmap64(addr, size + (ucs_memtrack_is_enabled() ? sizeof(*buffer) : 0),
                    prot, flags, fd, offset);
    if ((buffer == MAP_FAILED) || (!ucs_memtrack_is_enabled())) {
        return buffer;
    }

    if (fd > 0) {
        memmove(buffer + 1, buffer, size);
    }

    ucs_memtrack_record_alloc(buffer, size, 0, name);
    return buffer + 1;
}
#endif

int ucs_munmap(void *addr, size_t length)
{
    ucs_memtrack_buffer_t *buffer;

    if (!ucs_memtrack_is_enabled()) {
        return munmap(addr, length);
    }

    buffer = (ucs_memtrack_buffer_t*)addr - 1;
    ucs_memtrack_record_release(buffer, length);
    return munmap((void*)buffer - buffer->offset,
                  length + sizeof(*buffer) + buffer->offset);
}

char *ucs_strdup(const char *src, const char *name)
{
    char *str;
    size_t len = strlen(src);

    str = ucs_malloc(len + 1, name);
    if (str) {
        memcpy(str, src, len + 1);
    }

    return str;
}

static unsigned ucs_memtrack_total_internal(ucs_memtrack_entry_t* total)
{
    struct sglib_hashed_ucs_memtrack_entry_t_iterator entry_it;
    ucs_memtrack_entry_t *entry;
    unsigned num_entries;

    ucs_memtrack_total_reset(total);

    num_entries          = 0;
    for (entry = sglib_hashed_ucs_memtrack_entry_t_it_init(&entry_it,
                                                           ucs_memtrack_context.entries);
         entry != NULL;
         entry = sglib_hashed_ucs_memtrack_entry_t_it_next(&entry_it))
    {
        total->size          += entry->size;
        total->peak_size     += entry->peak_size;
        total->count         += entry->count;
        total->peak_count    += entry->peak_count;
        ++num_entries;
    }
    return num_entries;
}

void ucs_memtrack_total(ucs_memtrack_entry_t* total)
{
    if (!ucs_memtrack_is_enabled()) {
        return;
    }

    pthread_mutex_lock(&ucs_memtrack_context.lock);
    ucs_memtrack_total_internal(total);
    pthread_mutex_unlock(&ucs_memtrack_context.lock);
}

static int ucs_memtrack_cmp_entries(const void *ptr1, const void *ptr2)
{
    const ucs_memtrack_entry_t *e1 = ptr1;
    const ucs_memtrack_entry_t *e2 = ptr2;

    return (int)((ssize_t)e2->peak_size - (ssize_t)e1->peak_size);
}

static void ucs_memtrack_dump_internal(FILE* output_stream)
{
    struct sglib_hashed_ucs_memtrack_entry_t_iterator entry_it;
    ucs_memtrack_entry_t *entry, *all_entries;
    ucs_memtrack_entry_t total = {"", 0};
    unsigned num_entries, i;

    if (!ucs_memtrack_is_enabled()) {
        return;
    }

    num_entries = ucs_memtrack_total_internal(&total);

    fprintf(output_stream, "%31s current / peak  %16s current / peak\n", "", "");
    fprintf(output_stream, UCS_MEMTRACK_FORMAT_STRING, "TOTAL",
            total.size, total.peak_size,
            total.count, total.peak_count);

    all_entries = malloc(sizeof(ucs_memtrack_entry_t) * num_entries);

    /* Copy all entries to one array */
    i = 0;
    for (entry = sglib_hashed_ucs_memtrack_entry_t_it_init(&entry_it,
                                                           ucs_memtrack_context.entries);
         entry != NULL;
         entry = sglib_hashed_ucs_memtrack_entry_t_it_next(&entry_it))
    {
        all_entries[i++] = *entry;
    }
    ucs_assert(i == num_entries);

    /* Sort the entries from large to small */
    qsort(all_entries, num_entries, sizeof(ucs_memtrack_entry_t), ucs_memtrack_cmp_entries);
    for (i = 0; i < num_entries; ++i) {
        entry = &all_entries[i];
        fprintf(output_stream, UCS_MEMTRACK_FORMAT_STRING, entry->name,
                entry->size, entry->peak_size, entry->count, entry->peak_count);
    }

    free(all_entries);
}

void ucs_memtrack_dump(FILE* output_stream)
{
    pthread_mutex_lock(&ucs_memtrack_context.lock);
    ucs_memtrack_dump_internal(output_stream);
    pthread_mutex_unlock(&ucs_memtrack_context.lock);
}

static void ucs_memtrack_generate_report()
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

void ucs_memtrack_init()
{
    ucs_status_t status;

    ucs_assert(ucs_memtrack_context.enabled == 0);

    if (!strcmp(ucs_global_opts.memtrack_dest, "")) {
        ucs_trace("memtrack disabled");
        ucs_memtrack_context.enabled = 0;
        return;
    }

    sglib_hashed_ucs_memtrack_entry_t_init(ucs_memtrack_context.entries);
    status = UCS_STATS_NODE_ALLOC(&ucs_memtrack_context.stats,
                                  &ucs_memtrack_stats_class,
                                  ucs_stats_get_root());
    if (status != UCS_OK) {
        return;
    }

    ucs_debug("memtrack enabled");
    ucs_memtrack_context.enabled = 1;
}

void ucs_memtrack_cleanup()
{
    struct sglib_hashed_ucs_memtrack_entry_t_iterator entry_it;
    ucs_memtrack_entry_t *entry;

    if (!ucs_memtrack_context.enabled) {
        return;
    }

    pthread_mutex_lock(&ucs_memtrack_context.lock);

    ucs_memtrack_generate_report();

    /* disable before releasing the stats node */
    ucs_memtrack_context.enabled = 0;
    UCS_STATS_NODE_FREE(ucs_memtrack_context.stats);
    for (entry = sglib_hashed_ucs_memtrack_entry_t_it_init(&entry_it,
                                                           ucs_memtrack_context.entries);
         entry != NULL;
         entry = sglib_hashed_ucs_memtrack_entry_t_it_next(&entry_it))
    {
        sglib_hashed_ucs_memtrack_entry_t_delete(ucs_memtrack_context.entries, entry);
        free(entry);
    }
    pthread_mutex_unlock(&ucs_memtrack_context.lock);
}

int ucs_memtrack_is_enabled()
{
    return ucs_memtrack_context.enabled;
}

size_t ucs_memtrack_adjust_alloc_size(size_t size)
{
    return size + sizeof(ucs_memtrack_buffer_t);
}

void ucs_memtrack_allocated(void **ptr_p, size_t *size_p, const char *name)
{
    ucs_memtrack_buffer_t *buffer;

    if (!ucs_memtrack_is_enabled()) {
        return;
    }

    buffer   = *ptr_p;
    *ptr_p   = buffer + 1;
    *size_p -= sizeof(*buffer);
    ucs_memtrack_record_alloc(buffer, *size_p, 0, name);
}

void ucs_memtrack_releasing(void **ptr_p)
{
    ucs_memtrack_buffer_t *buffer;

    if (!ucs_memtrack_is_enabled()) {
        return;
    }

    buffer = *ptr_p -= sizeof(*buffer);
    ucs_memtrack_record_release(buffer, 0);
}

void ucs_memtrack_releasing_adjusted(void *ptr)
{
    ucs_memtrack_record_release(ptr, 0);
}

static uint64_t ucs_memtrack_entry_hash(ucs_memtrack_entry_t *entry)
{
    return ucs_string_to_id(entry->name);
}

static int ucs_memtrack_entry_compare(ucs_memtrack_entry_t *entry1,
                                      ucs_memtrack_entry_t *entry2)
{
    return strcmp(entry1->name, entry2->name);
}

SGLIB_DEFINE_LIST_FUNCTIONS(ucs_memtrack_entry_t, ucs_memtrack_entry_compare, next)
SGLIB_DEFINE_HASHED_CONTAINER_FUNCTIONS(ucs_memtrack_entry_t,
                                        UCS_MEMTRACK_ENTRY_HASH_SIZE,
                                        ucs_memtrack_entry_hash)

#endif


void ucs_memtrack_total_reset(ucs_memtrack_entry_t* total)
{
    ucs_snprintf_zero(total->name, UCS_MEMTRACK_NAME_MAX, "total");
    total->size          = 0;
    total->peak_size     = 0;
    total->count         = 0;
    total->peak_count    = 0;
}
