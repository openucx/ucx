/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "profile.h"

#include <ucs/datastruct/list.h>
#include <ucs/debug/debug_int.h>
#include <ucs/debug/log.h>
#include <ucs/sys/lib.h>
#include <ucs/sys/string.h>
#include <ucs/sys/sys.h>
#include <ucs/time/time.h>
#include <pthread.h>


typedef struct ucs_profile_global_location {
    ucs_profile_location_t       super;      /*< Location info */
    volatile int                 *loc_id_p;  /*< Back-pointer to location index */
} ucs_profile_global_location_t;


/* Profiling per-thread context */
typedef struct ucs_profile_thread_context {
    pthread_t                         pthread_id;    /**< POSIX thread id */
    int                               tid;           /**< System thread id */
    ucs_time_t                        start_time;    /**< Thread context init time */
    ucs_time_t                        end_time;      /**< Thread end time */
    ucs_list_link_t                   list;          /**< Entry in thread list */
    int                               is_completed;  /**< Set to 1 when thread exits */

    struct {
        ucs_profile_record_t          *start;        /**< Circular log buffer start */
        ucs_profile_record_t          *end;          /**< Circular log buffer end */
        ucs_profile_record_t          *current;      /**< Current log pointer */
        int                           wraparound;    /**< Whether log was rotated */
    } log;

    struct {
        unsigned                      num_locations; /**< Number of valid locations */
        ucs_profile_thread_location_t *locations;    /**< Statistics per location */
        int                           stack_top;     /**< Index of stack top */
        ucs_time_t                    stack[UCS_PROFILE_STACK_MAX]; /**< Timestamps for each nested scope */
    } accum;
} ucs_profile_thread_context_t;


/**
 * Profiling context
 */
struct ucs_profile_context  {
    unsigned                      profile_mode;     /**< Profiling mode */
    const char                    *file_name;       /**< Profiling output file name */
    size_t                        max_file_size;    /**< Limit for profiling log size */
    ucs_profile_global_location_t *locations;       /**< Array of all locations */
    unsigned                      num_locations;    /**< Number of valid locations */
    unsigned                      max_locations;    /**< Size of locations array */
    pthread_mutex_t               mutex;            /**< Protects updating the locations array */
    pthread_key_t                 tls_key;          /**< TLS key for per-thread context */
    ucs_list_link_t               thread_list;      /**< List of all thread contexts */
};


#define ucs_profile_ctx_for_each_location(_ctx, _var) \
    for ((_var) = (_ctx)->locations; \
         (_var) < ((_ctx)->locations + \
                   (_ctx)->num_locations); \
         ++(_var))


const char *ucs_profile_mode_names[] = {
    [UCS_PROFILE_MODE_ACCUM] = "accum",
    [UCS_PROFILE_MODE_LOG]   = "log",
    [UCS_PROFILE_MODE_LAST]  = NULL
};

/**
 *  Default ucs profile context is initialized in ucs_init() and used by
 *  UCS_PROFILE_ macros
 */
ucs_profile_context_t *ucs_profile_default_ctx;

static ucs_status_t ucs_profile_file_write_data(int fd, void *data, size_t size)
{
    ssize_t written;

    if (size > 0) {
        written = write(fd, data, size);
        if (written < 0) {
            ucs_error("failed to write %zu bytes to profiling file: %m", size);
            return UCS_ERR_IO_ERROR;
        } else if (size != written) {
            ucs_error("wrote only %zd of %zu bytes to profiling file: %m",
                      written, size);
            return UCS_ERR_IO_ERROR;
        }
    }

    return UCS_OK;
}

static ucs_status_t
ucs_profile_file_write_records(int fd, ucs_profile_record_t *begin,
                               ucs_profile_record_t *end)
{
    return ucs_profile_file_write_data(fd, begin, UCS_PTR_BYTE_DIFF(begin, end));
}

/* Global lock must be held */
static ucs_status_t
ucs_profile_file_write_thread(ucs_profile_context_t *ctx, int fd,
                              ucs_profile_thread_context_t *thread_ctx,
                              ucs_time_t default_end_time)
{
    ucs_profile_thread_location_t empty_location = { .total_time = 0, .count = 0 };
    ucs_profile_thread_header_t thread_hdr;
    unsigned i, num_locations;
    ucs_status_t status;

    /*
     * NOTE: There is no protection against a race with a thread which is still
     * producing profiling data (e.g updating the context structure without a
     * lock).
     * To avoid excess locking on fast-path, we assume that when we dump the
     * profiling data (at program exit), the profiled threads are not calling
     * ucs_profile_record() anymore.
     */

    ucs_debug("profiling thread context %p: write to file", thread_ctx);

    /* write thread header */
    thread_hdr.tid          = thread_ctx->tid;
    thread_hdr.start_time   = thread_ctx->start_time;
    if (thread_ctx->is_completed) {
        thread_hdr.end_time = thread_ctx->end_time;
    } else {
        thread_hdr.end_time = default_end_time;
    }

    if (ctx->profile_mode & UCS_BIT(UCS_PROFILE_MODE_LOG)) {
        thread_hdr.num_records = thread_ctx->log.wraparound ?
                             (thread_ctx->log.end     - thread_ctx->log.start) :
                             (thread_ctx->log.current - thread_ctx->log.start);
    } else {
        thread_hdr.num_records = 0;
    }

    status = ucs_profile_file_write_data(fd, &thread_hdr, sizeof(thread_hdr));
    if (status != UCS_OK) {
        return status;
    }

    /* If accumulate mode is not enabled, there are no location entries */
    if (ctx->profile_mode & UCS_BIT(UCS_PROFILE_MODE_ACCUM)) {
        num_locations = thread_ctx->accum.num_locations;
    } else {
        num_locations = 0;
    }

    /* write profiling information for every location
     * note: the thread location array may be smaller (or even empty) than the
     * global list, but it cannot be larger. If it's smaller, we pad with empty
     * entries
     */
    ucs_assert_always(num_locations <= ctx->num_locations);
    ucs_profile_file_write_data(fd, thread_ctx->accum.locations,
                                num_locations * sizeof(*thread_ctx->accum.locations));
    for (i = num_locations; i < ctx->num_locations; ++i) {
        status = ucs_profile_file_write_data(fd, &empty_location,
                                             sizeof(empty_location));
        if (status != UCS_OK) {
            return status;
        }
    }

    /* write profiling records */
    if (ctx->profile_mode & UCS_BIT(UCS_PROFILE_MODE_LOG)) {
        if (thread_ctx->log.wraparound) {
            status = ucs_profile_file_write_records(fd, thread_ctx->log.current,
                                                    thread_ctx->log.end);
            if (status != UCS_OK) {
                return status;
            }
        }

        status = ucs_profile_file_write_records(fd, thread_ctx->log.start,
                                                thread_ctx->log.current);
        if (status != UCS_OK) {
            return status;
        }
    }

    return UCS_OK;
}

static ucs_status_t ucs_profile_write_locations(ucs_profile_context_t *ctx,
                                                int fd)
{
    ucs_profile_global_location_t *loc;
    ucs_status_t status;

    ucs_profile_ctx_for_each_location(ctx, loc) {
        status = ucs_profile_file_write_data(fd, &loc->super, sizeof(loc->super));
        if (status != UCS_OK) {
            return status;
        }
    }

    return UCS_OK;
}

static void ucs_profile_write(ucs_profile_context_t *ctx)
{
    ucs_profile_thread_context_t *thread_ctx;
    ucs_profile_header_t header;
    char fullpath[1024] = {0};
    char filename[1024] = {0};
    ucs_time_t write_time;
    ucs_status_t status;
    int fd;

    if (!ctx->profile_mode) {
        return;
    }

    pthread_mutex_lock(&ctx->mutex);

    write_time = ucs_get_time();

    ucs_fill_filename_template(ctx->file_name, filename, sizeof(filename));
    ucs_expand_path(filename, fullpath, sizeof(fullpath) - 1);

    fd = open(fullpath, O_WRONLY|O_CREAT|O_TRUNC, 0600);
    if (fd < 0) {
        ucs_error("failed to write profiling data to '%s': %m", fullpath);
        goto out_unlock;
    }

    /* write header */
    memset(&header, 0, sizeof(header));
    ucs_read_file(header.cmdline, sizeof(header.cmdline), 1, "/proc/self/cmdline");
    strncpy(header.hostname, ucs_get_host_name(), sizeof(header.hostname) - 1);
    header.version       = UCS_PROFILE_FILE_VERSION;
    strncpy(header.ucs_path, ucs_sys_get_lib_path(), sizeof(header.ucs_path) - 1);
    header.pid           = getpid();
    header.mode          = ctx->profile_mode;
    header.num_locations = ctx->num_locations;
    header.num_threads   = ucs_list_length(&ctx->thread_list);
    header.one_second    = ucs_time_from_sec(1.0);
    ucs_profile_file_write_data(fd, &header, sizeof(header));

    /* write locations */
    status = ucs_profile_write_locations(ctx, fd);
    if (status != UCS_OK) {
        goto out_close_fd;
    }

    /* write threads */
    ucs_list_for_each(thread_ctx, &ctx->thread_list, list) {
        status = ucs_profile_file_write_thread(ctx, fd, thread_ctx, write_time);
        if (status != UCS_OK) {
            goto out_close_fd;
        }
    }

out_close_fd:
    close(fd);
out_unlock:
    pthread_mutex_unlock(&ctx->mutex);
}

static UCS_F_NOINLINE ucs_profile_thread_context_t*
ucs_profile_thread_init(ucs_profile_context_t *ctx)
{
    ucs_profile_thread_context_t *thread_ctx;
    size_t num_records;

    ucs_assert(ctx->profile_mode);

    thread_ctx = ucs_malloc(sizeof(*thread_ctx), "profile_thread_context");
    if (thread_ctx == NULL) {
        ucs_error("failed to allocate profiling thread context");
        return NULL;
    }

    thread_ctx->tid        = ucs_get_tid();
    thread_ctx->start_time = ucs_get_time();
    thread_ctx->end_time   = 0;
    thread_ctx->pthread_id = pthread_self();

    ucs_debug("profiling context %p: start on thread 0x%lx tid %d mode %d",
              thread_ctx, (unsigned long)pthread_self(), ucs_get_tid(),
              ctx->profile_mode);

    /* Initialize log mode */
    if (ctx->profile_mode & UCS_BIT(UCS_PROFILE_MODE_LOG)) {
        num_records = ctx->max_file_size / sizeof(ucs_profile_record_t);
        thread_ctx->log.start = ucs_calloc(num_records,
                                           sizeof(ucs_profile_record_t),
                                           "profile_log");
        if (thread_ctx->log.start == NULL) {
            ucs_fatal("failed to allocate profiling log");
        }

        thread_ctx->log.end        = thread_ctx->log.start + num_records;
        thread_ctx->log.current    = thread_ctx->log.start;
        thread_ctx->log.wraparound = 0;
    }

    /* Initialize accumulate mode */
    if (ctx->profile_mode & UCS_BIT(UCS_PROFILE_MODE_ACCUM)) {
        thread_ctx->accum.num_locations = 0;
        thread_ctx->accum.locations     = NULL;
        thread_ctx->accum.stack_top     = -1;
    }

    pthread_setspecific(ctx->tls_key, thread_ctx);

    pthread_mutex_lock(&ctx->mutex);
    ucs_list_add_tail(&ctx->thread_list, &thread_ctx->list);
    pthread_mutex_unlock(&ctx->mutex);

    return thread_ctx;
}

static void ucs_profile_thread_cleanup(unsigned profile_mode,
                                       ucs_profile_thread_context_t *ctx)
{
    ucs_debug("profiling context %p: cleanup", ctx);

    if (profile_mode & UCS_BIT(UCS_PROFILE_MODE_LOG)) {
        ucs_free(ctx->log.start);
    }

    if (profile_mode & UCS_BIT(UCS_PROFILE_MODE_ACCUM)) {
        ucs_free(ctx->accum.locations);
    }

    ucs_list_del(&ctx->list);
    ucs_free(ctx);
}

static void ucs_profile_thread_finalize(ucs_profile_thread_context_t *ctx)
{
    ucs_debug("profiling context %p: completed", ctx);

    ctx->end_time     = ucs_get_time();
    ctx->is_completed = 1;
}

static void ucs_profile_thread_key_destr(void *data)
{
    ucs_profile_thread_context_t *ctx = data;
    ucs_profile_thread_finalize(ctx);
}

/*
 * Register a profiling location - should be called once per location in the
 * code, before the first record of each such location is made.
 * SHOULD NOT be used directly - use UCS_PROFILE macros instead.
 *
 * @param [in]  ctx          Profile context.
 * @param [in]  type         Location type.
 * @param [in]  name         Location name.
 * @param [in]  file         Source file name.
 * @param [in]  line         Source line number.
 * @param [in]  function     Calling function name.
 * @param [out] loc_id_p     Filled with location ID:
 *                             0   - profiling is disabled
 *                             >0  - location index + 1
 */
static UCS_F_NOINLINE int
ucs_profile_get_location(ucs_profile_context_t *ctx, ucs_profile_type_t type,
                         const char *name, const char *file, int line,
                         const char *function, volatile int *loc_id_p)
{
    ucs_profile_global_location_t *loc, *new_locations;
    int loc_id;

    pthread_mutex_lock(&ctx->mutex);

    /* Check, with lock held, that the location is not already initialized */
    if (*loc_id_p >= 0) {
        loc_id = *loc_id_p;
        goto out_unlock;
    }

    /* Check if profiling is disabled */
    if (!ctx->profile_mode) {
        *loc_id_p = loc_id = 0;
        goto out_unlock;
    }

    /* Location ID must be uninitialized */
    ucs_assert(*loc_id_p == -1);

    ucs_profile_ctx_for_each_location(ctx, loc) {
        if ((type == loc->super.type) && (line == loc->super.line) &&
            !strcmp(loc->super.name, name) &&
            !strcmp(loc->super.file, ucs_basename(file)) &&
            !strcmp(loc->super.function, function)) {
            goto out_found;
        }
    }

    ++(ctx->num_locations);

    /* Reallocate array if needed */
    if (ctx->num_locations > ctx->max_locations) {
        ctx->max_locations = 2 * ctx->num_locations;
        new_locations = ucs_realloc(ctx->locations,
                                    sizeof(*ctx->locations) * ctx->max_locations,
                                    "profile_locations");
        if (new_locations == NULL) {
            ucs_warn("failed to expand locations array");
            *loc_id_p = loc_id = 0;
            goto out_unlock;
        }

        ctx->locations = new_locations;
    }

    /* Initialize new location */
    loc = &(ctx->locations[ctx->num_locations - 1]);
    ucs_strncpy_zero(loc->super.file, ucs_basename(file), sizeof(loc->super.file));
    ucs_strncpy_zero(loc->super.function, function, sizeof(loc->super.function));
    ucs_strncpy_zero(loc->super.name, name, sizeof(loc->super.name));
    loc->super.line = line;
    loc->super.type = type;
    loc->loc_id_p   = loc_id_p;

out_found:
    *loc_id_p = loc_id = (loc - ctx->locations) + 1;
    ucs_memory_cpu_store_fence();
out_unlock:
    pthread_mutex_unlock(&ctx->mutex);
    return loc_id;
}

static void ucs_profile_thread_expand_locations(ucs_profile_context_t *ctx,
                                                int loc_id)
{
    ucs_profile_thread_context_t *thread_ctx;
    unsigned i, new_num_locations;

    thread_ctx = pthread_getspecific(ctx->tls_key);
    ucs_assert(thread_ctx != NULL);

    new_num_locations = ucs_max(loc_id, thread_ctx->accum.num_locations);
    thread_ctx->accum.locations = ucs_realloc(thread_ctx->accum.locations,
                                       sizeof(*thread_ctx->accum.locations) *
                                       new_num_locations,
                                       "profile_thread_locations");
    if (thread_ctx->accum.locations == NULL) {
        ucs_fatal("failed to allocate profiling per-thread locations");
    }

    for (i = thread_ctx->accum.num_locations; i < new_num_locations; ++i) {
        thread_ctx->accum.locations[i].count      = 0;
        thread_ctx->accum.locations[i].total_time = 0;
    }

    thread_ctx->accum.num_locations = new_num_locations;
}

void ucs_profile_record(ucs_profile_context_t *ctx, ucs_profile_type_t type,
                        const char *name, uint32_t param32, uint64_t param64,
                        const char *file, int line, const char *function,
                        volatile int *loc_id_p)
{
    ucs_profile_thread_location_t *loc;
    ucs_profile_thread_context_t *thread_ctx;
    ucs_profile_record_t *rec;
    ucs_time_t current_time;
    int loc_id;

    /* If the location id is -1 or 0, need to re-read it with lock held */
    loc_id = *loc_id_p;
    if (ucs_unlikely(loc_id <= 0)) {
        loc_id = ucs_profile_get_location(ctx, type, name, file, line,
                                          function, loc_id_p);
        if (loc_id == 0) {
            return;
        }
    }

    ucs_memory_cpu_load_fence();

    ucs_assert(*loc_id_p            != 0);
    ucs_assert(ctx->profile_mode != 0);

    /* Get thread-specific profiling context */
    thread_ctx = pthread_getspecific(ctx->tls_key);
    if (ucs_unlikely(thread_ctx == NULL)) {
        thread_ctx = ucs_profile_thread_init(ctx);
    }

    current_time = ucs_get_time();
    if (ctx->profile_mode & UCS_BIT(UCS_PROFILE_MODE_ACCUM)) {
        if (ucs_unlikely(loc_id > thread_ctx->accum.num_locations)) {
            /* expand the locations array of the current thread */
            ucs_profile_thread_expand_locations(ctx, loc_id);
        }
        ucs_assert(loc_id - 1 < thread_ctx->accum.num_locations);

        loc = &thread_ctx->accum.locations[loc_id - 1];
        switch (type) {
        case UCS_PROFILE_TYPE_SCOPE_BEGIN:
            thread_ctx->accum.stack[++thread_ctx->accum.stack_top] = current_time;
            break;
        case UCS_PROFILE_TYPE_SCOPE_END:
            loc->total_time += current_time -
                               thread_ctx->accum.stack[thread_ctx->accum.stack_top];
            --thread_ctx->accum.stack_top;
            break;
        default:
            break;
        }
        ++loc->count;
    }

    if (ctx->profile_mode & UCS_BIT(UCS_PROFILE_MODE_LOG)) {
        rec              = thread_ctx->log.current;
        rec->timestamp   = current_time;
        rec->param64     = param64;
        rec->param32     = param32;
        rec->location    = loc_id - 1;
        if (++thread_ctx->log.current >= thread_ctx->log.end) {
            thread_ctx->log.current    = thread_ctx->log.start;
            thread_ctx->log.wraparound = 1;
        }
    }
}

static void ucs_profile_check_active_threads(ucs_profile_context_t *ctx)
{
    size_t num_active_threads;

    pthread_mutex_lock(&ctx->mutex);
    num_active_threads = ucs_list_length(&ctx->thread_list);
    pthread_mutex_unlock(&ctx->mutex);

    if (num_active_threads > 0) {
        ucs_warn("%zd profiled threads are still running", num_active_threads);
    }
}

void ucs_profile_reset_locations_id(ucs_profile_context_t *ctx)
{
    ucs_profile_global_location_t *loc;

    pthread_mutex_lock(&ctx->mutex);

    ucs_profile_ctx_for_each_location(ctx, loc) {
        *loc->loc_id_p = -1;
    }

    pthread_mutex_unlock(&ctx->mutex);
}

static void ucs_profile_reset_locations(ucs_profile_context_t *ctx)
{
    pthread_mutex_lock(&ctx->mutex);

    ctx->num_locations = 0;
    ctx->max_locations = 0;
    ucs_free(ctx->locations);
    ctx->locations = NULL;

    pthread_mutex_unlock(&ctx->mutex);
}

static void ucs_profile_cleanup_completed_threads(ucs_profile_context_t *ctx)
{
    ucs_profile_thread_context_t *thread_ctx, *tmp;

    pthread_mutex_lock(&ctx->mutex);
    ucs_list_for_each_safe(thread_ctx, tmp, &ctx->thread_list, list) {
        if (thread_ctx->is_completed) {
            ucs_profile_thread_cleanup(ctx->profile_mode, thread_ctx);
        }
    }
    pthread_mutex_unlock(&ctx->mutex);
}

void ucs_profile_dump(ucs_profile_context_t *ctx)
{
    ucs_profile_thread_context_t *thread_ctx;

    /* finalize profiling on current thread */
    thread_ctx = pthread_getspecific(ctx->tls_key);
    if (thread_ctx != NULL) {
        ucs_profile_thread_finalize(thread_ctx);
        pthread_setspecific(ctx->tls_key, NULL);
    }

    /* write and cleanup all completed threads (including the current thread) */
    ucs_profile_write(ctx);
    ucs_profile_cleanup_completed_threads(ctx);
}

ucs_status_t ucs_profile_init(unsigned profile_mode, const char *file_name,
                              size_t max_file_size, ucs_profile_context_t **ctx_p)
{
    ucs_profile_context_t *ctx;
    ucs_status_t status;
    int ret;

    ctx = ucs_malloc(sizeof(*ctx), "ucs profile context");
    if (ctx == NULL) {
        ucs_error("failed to allocate memory for ucs_profile_context_t");
        return UCS_ERR_NO_MEMORY;
    }

    ret = pthread_mutex_init(&ctx->mutex, NULL);
    if (ret != 0) {
        ucs_error("failed to initialize mutex");
        status = UCS_ERR_IO_ERROR;
        goto free_ctx;
    }

    ucs_list_head_init(&ctx->thread_list);
    ctx->profile_mode     = profile_mode;
    ctx->file_name        = file_name;
    ctx->max_file_size    = max_file_size;
    /* coverity[missing_lock] */
    ctx->num_locations    = 0;
    ctx->locations        = NULL;
    ctx->max_locations    = 0;

    if (profile_mode && !strlen(file_name)) {
        // TODO make sure profiling file is writeable
        ucs_warn("profiling file not specified");
    }

    pthread_key_create(&(ctx->tls_key), ucs_profile_thread_key_destr);
    *ctx_p = ctx;

    return UCS_OK;

free_ctx:
    ucs_free(ctx);
    return status;
}

void ucs_profile_cleanup(ucs_profile_context_t *ctx)
{
    ucs_profile_dump(ctx);
    ucs_profile_check_active_threads(ctx);
    ucs_profile_reset_locations(ctx);
    pthread_key_delete(ctx->tls_key);
    ucs_free(ctx);
}
