/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2013.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

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

#if ENABLE_MEMTRACK

#define MEMTRACK_BUFFER_MAGIC        0x1ee7beefa880feed
#define MEMTRACK_DUMP_FORMAT_STRING  ("%22s: size: %9lu / %9lu\tcount: %9lu / %9lu\n")
#define UCS_MEMTRACK_ENTRY_HASH_SIZE 127


typedef struct ucs_memtrack_context {
    int                  enabled;
    pthread_mutex_t      lock;
    ucs_memtrack_entry_t *entries[UCS_MEMTRACK_ENTRY_HASH_SIZE];
    UCS_STATS_NODE_DECLARE(stats);
} ucs_memtrack_context_t;


/* Global context for tracking all the memory */
static ucs_memtrack_context_t ucs_memtrack_context = {
    .enabled = 0,
    .lock    = PTHREAD_MUTEX_INITIALIZER
};


static inline unsigned ucs_memtrack_entry_hash(ucs_memtrack_entry_t *entry)
{
    return entry->origin % UCS_MEMTRACK_ENTRY_HASH_SIZE;
}

static inline int ucs_memtrack_entry_compare(ucs_memtrack_entry_t *entry1,
                                             ucs_memtrack_entry_t *entry2)
{
    if (entry1->origin != entry2->origin) {
        return (int)entry1->origin - (int)entry2->origin;
    } else {
        return strcmp(entry1->alloc_name, entry2->alloc_name);
    }
}


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

static inline ucs_memtrack_entry_t* ucs_memtrack_create_entry(const char* alloc_name,
                                                              unsigned origin)
{
    ucs_memtrack_entry_t *new_entry = malloc(sizeof(*new_entry));
    if (new_entry == NULL) {
        return NULL;
    }

    new_entry->current_size = 0;
    new_entry->peak_size = 0;
    new_entry->current_count = 0;
    new_entry->peak_count = 0;

    new_entry->origin = origin;
    new_entry->alloc_name = alloc_name ? strdup(alloc_name) : NULL;
    sglib_hashed_ucs_memtrack_entry_t_add(ucs_memtrack_context.entries, new_entry);
    return new_entry;
}

void ucs_memtrack_record_alloc(ucs_memtrack_buffer_t* buffer, size_t size
                               UCS_MEMTRACK_ARG)
{
    ucs_memtrack_entry_t *new_entry, search = {0};

    if (!ucs_memtrack_context.enabled) {
        return;
    }

    ucs_assert(buffer != NULL);
    ucs_assert(alloc_name != NULL);
    ucs_assert(ucs_memtrack_context.entries != NULL); // context initialized
    pthread_mutex_lock(&ucs_memtrack_context.lock);

    search.origin = origin;
    search.alloc_name = (char*)((void*)alloc_name);
    new_entry = sglib_hashed_ucs_memtrack_entry_t_find_member(ucs_memtrack_context.entries,
                                                              &search);
    if (new_entry == NULL) {
        new_entry = ucs_memtrack_create_entry(alloc_name, origin);
        if (new_entry == NULL) {
            goto out_unlock;
        }
    }

    ucs_assert(0 == strcmp(alloc_name, new_entry->alloc_name));
    buffer->magic = MEMTRACK_BUFFER_MAGIC;
    buffer->length = size;
    buffer->offset = 0;
    buffer->entry = new_entry;

    new_entry->current_size += size;
    UCS_STATS_UPDATE_COUNTER(ucs_memtrack_context.stats, UCS_MEMTRACK_STAT_ALLOCATION_SIZE, size);
    new_entry->peak_size = ucs_max(new_entry->peak_size, new_entry->current_size);

    new_entry->current_count++;
    UCS_STATS_UPDATE_COUNTER(ucs_memtrack_context.stats, UCS_MEMTRACK_STAT_ALLOCATION_COUNT, 1);
    new_entry->peak_count = ucs_max(new_entry->peak_count, new_entry->current_count);

out_unlock:
    pthread_mutex_unlock(&ucs_memtrack_context.lock);
}

ucs_memtrack_entry_t* ucs_memtrack_record_dealloc(ucs_memtrack_buffer_t *buffer)
{
    ucs_memtrack_entry_t *res;
    if (!ucs_memtrack_context.enabled) {
        return NULL;
    }

    pthread_mutex_lock(&ucs_memtrack_context.lock);
    ucs_assert(buffer->magic == MEMTRACK_BUFFER_MAGIC);
#if ENABLE_ASSERT
    buffer->magic = MEMTRACK_BUFFER_MAGIC + 1; /* protect from double free */
#endif

    res = buffer->entry;
    ucs_assert(res->current_size >= buffer->length);
    res->current_size -= buffer->length;
    ucs_assert(res->current_count >= 1);
    res->current_count--;

    pthread_mutex_unlock(&ucs_memtrack_context.lock);
    return res;
}

void *ucs_memtrack_calloc(size_t nmemb, size_t size UCS_MEMTRACK_ARG)
{
    ucs_memtrack_buffer_t *res;
    res = calloc(1, nmemb * size + (ucs_memtrack_context.enabled ? sizeof(*res) : 0));
    if ((res == NULL) || (!ucs_memtrack_context.enabled)) {
        return res;
    }

    ucs_memtrack_record_alloc(res, nmemb * size, alloc_name, origin);
    return res + 1;
}

void *ucs_memtrack_malloc(size_t size UCS_MEMTRACK_ARG)
{
    ucs_memtrack_buffer_t *res;
    res = malloc(size + (ucs_memtrack_context.enabled ? sizeof(*res) : 0));
    if ((res == NULL) || (!ucs_memtrack_context.enabled)) {
        return res;
    }

    ucs_memtrack_record_alloc(res, size, alloc_name, origin);
    return res + 1;
}

void *ucs_memtrack_memalign(size_t boundary, size_t size UCS_MEMTRACK_ARG)
{
    size_t offset = 0;
    ucs_memtrack_buffer_t *res;
    if (!ucs_memtrack_context.enabled) {
        return memalign(boundary, size);
    }

    if (boundary > sizeof(*res)) {
        res = memalign(boundary, size + boundary);
        offset = boundary - sizeof(*res);
    } else {
        if (sizeof(*res) % boundary != 0) {
            offset = boundary - (sizeof(*res) % boundary);
        }
        res = memalign(boundary, size + sizeof(*res) + offset);
    }
    if ((res == NULL) || (!ucs_memtrack_context.enabled)) {
        return res;
    }

    res = (void*)res + offset;
    ucs_memtrack_record_alloc(res, size, alloc_name, origin);
    res->offset = offset;
    return res + 1;
}

void ucs_memtrack_free(void *ptr)
{
    ucs_memtrack_buffer_t *buffer = (ucs_memtrack_buffer_t*)ptr - 1;
    if (!ucs_memtrack_context.enabled) {
        free(ptr);
    } else {
        if (ptr == NULL) {
            return;
        }
        ucs_memtrack_record_dealloc(buffer);
        free((void*)buffer - buffer->offset);
    }
}

void *ucs_memtrack_mmap(void *addr, size_t length, int prot, int flags,
                        int fd, off_t offset UCS_MEMTRACK_ARG)
{
    ucs_memtrack_buffer_t *res;
    if ((flags & MAP_FIXED) || !(prot & PROT_WRITE)) {
        return NULL;
    }

    res = mmap(addr, length + (ucs_memtrack_context.enabled ? sizeof(*res) : 0),
               prot, flags, fd, offset);
    if ((res == NULL) || (!ucs_memtrack_context.enabled)) {
        return res;
    }

    if (fd > 0) {
        memmove(res + 1, res, length);
    }

    ucs_memtrack_record_alloc(res, length UCS_MEMTRACK_VAL);
    return res + 1;
}

#ifdef __USE_LARGEFILE64
void *ucs_memtrack_mmap64(void *addr, size_t length, int prot, int flags,
                          int fd, uint64_t offset UCS_MEMTRACK_ARG)
{
    ucs_memtrack_buffer_t *res;
    if ((flags & MAP_FIXED) || !(prot & PROT_WRITE)) {
        return NULL;
    }

    res = mmap64(addr, length + (ucs_memtrack_context.enabled ? sizeof(*res) : 0),
                 prot, flags, fd, offset);
    if ((res == NULL) || (!ucs_memtrack_context.enabled)) {
        return res;
    }

    if (fd > 0) {
        memmove(res + 1, res, length);
    }

    ucs_memtrack_record_alloc(res, length UCS_MEMTRACK_VAL);
    return res + 1;
}
#endif

int ucs_memtrack_munmap(void *addr, size_t length)
{
    ucs_memtrack_buffer_t *buffer = (ucs_memtrack_buffer_t*)addr - 1;
    if (!ucs_memtrack_context.enabled) {
        return munmap(addr, length);
    }

    ucs_assert(buffer->length == length); // Otherwise it's part of the buffer!
    ucs_memtrack_record_dealloc(buffer);
    return munmap((void*)buffer - buffer->offset,
                  length + sizeof(*buffer) + buffer->offset);
}

void *ucs_memtrack_realloc(void *ptr, size_t size)
{
    ucs_memtrack_buffer_t *buffer = (ucs_memtrack_buffer_t*)ptr - 1;
    ucs_memtrack_entry_t *entry;

    if (!ucs_memtrack_context.enabled) {
        return realloc(ptr, size);
    }

    entry = ucs_memtrack_record_dealloc(buffer);
    buffer = realloc((void*)buffer - buffer->offset, size + sizeof(*buffer));
    if (buffer == NULL) {
        return NULL;
    }
    if (!ucs_memtrack_context.enabled) {
        return buffer;
    }

    if (entry != NULL) {
        ucs_memtrack_record_alloc(buffer, size, entry->alloc_name,
                                    entry->origin);
    }
    return buffer + 1;
}

static unsigned ucs_memtrack_total_internal(ucs_memtrack_entry_t* total)
{
    struct sglib_hashed_ucs_memtrack_entry_t_iterator entry_it;
    ucs_memtrack_entry_t *entry;
    unsigned num_entries;

    if (!ucs_memtrack_context.enabled) {
        return 0;
    }

    num_entries          = 0;
    total->current_size  = 0;
    total->peak_size     = 0;
    total->current_count = 0;
    total->peak_count    = 0;

    for (entry = sglib_hashed_ucs_memtrack_entry_t_it_init(&entry_it,
                                                           ucs_memtrack_context.entries);
         entry != NULL;
         entry = sglib_hashed_ucs_memtrack_entry_t_it_next(&entry_it))
    {
        total->current_size  += entry->current_size;
        total->peak_size     += entry->peak_size;
        total->current_count += entry->current_count;
        total->peak_count    += entry->peak_count;
        ++num_entries;
    }

    return num_entries;
}

void ucs_memtrack_total(ucs_memtrack_entry_t* total)
{
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
    ucs_memtrack_entry_t total = {0};
    ucs_memtrack_entry_t *entry, *all_entries;
    unsigned num_entries, i;

    if (!ucs_memtrack_context.enabled) {
        return;
    }

    num_entries = ucs_memtrack_total_internal(&total);

    fprintf(output_stream, "%31s current / peak  %16s current / peak\n", "", "");
    fprintf(output_stream, MEMTRACK_DUMP_FORMAT_STRING, "TOTAL",
            total.current_size, total.peak_size,
            total.current_count, total.peak_count);

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
        fprintf(output_stream, MEMTRACK_DUMP_FORMAT_STRING, entry->alloc_name,
                entry->current_size, entry->peak_size,
                entry->current_count, entry->peak_count);
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

    status = ucs_open_output_stream(ucs_global_opts.memtrack_dest, &output_stream,
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
    status = UCS_STATS_NODE_ALLOC(&ucs_memtrack_context.stats, &ucs_memtrack_stats_class, NULL);
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
        free(entry->alloc_name);
        free(entry);
    }
    pthread_mutex_unlock(&ucs_memtrack_context.lock);
}

int ucs_memtrack_is_enabled()
{
    return ucs_memtrack_context.enabled;
}

SGLIB_DEFINE_LIST_FUNCTIONS(ucs_memtrack_entry_t, ucs_memtrack_entry_compare, next)
SGLIB_DEFINE_HASHED_CONTAINER_FUNCTIONS(ucs_memtrack_entry_t,
                                        UCS_MEMTRACK_ENTRY_HASH_SIZE,
                                        ucs_memtrack_entry_hash)

#else

void ucs_memtrack_init()
{
}

void ucs_memtrack_cleanup()
{
}

#endif
