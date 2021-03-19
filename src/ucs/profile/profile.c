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
#include <ucs/sys/string.h>
#include <ucs/sys/sys.h>
#include <ucs/time/time.h>
#include <pthread.h>


typedef struct ucs_profile_global_location {
    ucs_profile_location_t       super;      /*< Location info */
    volatile int                 *loc_id_p;  /*< Back-pointer to location index */
} ucs_profile_global_location_t;


/**
 * Profiling global context
 */
typedef struct ucs_profile_global_context {
    ucs_profile_global_location_t *locations;    /**< Array of all locations */
    unsigned                      num_locations; /**< Number of valid locations */
    unsigned                      max_locations; /**< Size of locations array */
    pthread_mutex_t               mutex;         /**< Protects updating the locations array */
    pthread_key_t                 tls_key;       /**< TLS key for per-thread context */
    ucs_list_link_t               thread_list;   /**< List of all thread contexts */
} ucs_profile_global_context_t;


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


#define ucs_profile_for_each_location(_var) \
    for ((_var) = ucs_profile_global_ctx.locations; \
         (_var) < (ucs_profile_global_ctx.locations + \
                   ucs_profile_global_ctx.num_locations); \
         ++(_var))


const char *ucs_profile_mode_names[] = {
    [UCS_PROFILE_MODE_ACCUM] = "accum",
    [UCS_PROFILE_MODE_LOG]   = "log",
    [UCS_PROFILE_MODE_LAST]  = NULL
};

static ucs_profile_global_context_t ucs_profile_global_ctx = {
    .locations     = NULL,
    .num_locations = 0,
    .max_locations = 0,
    .mutex         = PTHREAD_MUTEX_INITIALIZER,
    .thread_list   = UCS_LIST_INITIALIZER(&ucs_profile_global_ctx.thread_list,
                                          &ucs_profile_global_ctx.thread_list),
};

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
ucs_profile_file_write_thread(int fd, ucs_profile_thread_context_t *ctx,
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

    ucs_debug("profiling context %p: write to file", ctx);

    /* write thread header */
    thread_hdr.tid          = ctx->tid;
    thread_hdr.start_time   = ctx->start_time;
    if (ctx->is_completed) {
        thread_hdr.end_time = ctx->end_time;
    } else {
        thread_hdr.end_time = default_end_time;
    }

    if (ucs_global_opts.profile_mode & UCS_BIT(UCS_PROFILE_MODE_LOG)) {
        thread_hdr.num_records = ctx->log.wraparound ?
                             (ctx->log.end     - ctx->log.start) :
                             (ctx->log.current - ctx->log.start);
    } else {
        thread_hdr.num_records = 0;
    }

    status = ucs_profile_file_write_data(fd, &thread_hdr, sizeof(thread_hdr));
    if (status != UCS_OK) {
        return status;
    }

    /* If accumulate mode is not enabled, there are no location entries */
    if (ucs_global_opts.profile_mode & UCS_BIT(UCS_PROFILE_MODE_ACCUM)) {
        num_locations = ctx->accum.num_locations;
    } else {
        num_locations = 0;
    }

    /* write profiling information for every location
     * note: the thread location array may be smaller (or even empty) than the
     * global list, but it cannot be larger. If it's smaller, we pad with empty
     * entries
     */
    ucs_assert_always(num_locations <= ucs_profile_global_ctx.num_locations);
    ucs_profile_file_write_data(fd, ctx->accum.locations,
                                num_locations * sizeof(*ctx->accum.locations));
    for (i = num_locations; i < ucs_profile_global_ctx.num_locations; ++i) {
        status = ucs_profile_file_write_data(fd, &empty_location,
                                             sizeof(empty_location));
        if (status != UCS_OK) {
            return status;
        }
    }

    /* write profiling records */
    if (ucs_global_opts.profile_mode & UCS_BIT(UCS_PROFILE_MODE_LOG)) {
        if (ctx->log.wraparound) {
            status = ucs_profile_file_write_records(fd, ctx->log.current,
                                                    ctx->log.end);
            if (status != UCS_OK) {
                return status;
            }
        }

        status = ucs_profile_file_write_records(fd, ctx->log.start,
                                                ctx->log.current);
        if (status != UCS_OK) {
            return status;
        }
    }

    return UCS_OK;
}

static ucs_status_t ucs_profile_write_locations(int fd)
{
    ucs_profile_global_location_t *loc;
    ucs_status_t status;

    ucs_profile_for_each_location(loc) {
        status = ucs_profile_file_write_data(fd, &loc->super, sizeof(loc->super));
        if (status != UCS_OK) {
            return status;
        }
    }

    return UCS_OK;
}

static void ucs_profile_write()
{
    ucs_profile_thread_context_t *ctx;
    ucs_profile_header_t header;
    char fullpath[1024] = {0};
    char filename[1024] = {0};
    ucs_time_t write_time;
    ucs_status_t status;
    int fd;

    if (!ucs_global_opts.profile_mode) {
        return;
    }

    pthread_mutex_lock(&ucs_profile_global_ctx.mutex);

    write_time = ucs_get_time();

    ucs_fill_filename_template(ucs_global_opts.profile_file,
                               filename, sizeof(filename));
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
    strncpy(header.ucs_path, ucs_debug_get_lib_path(), sizeof(header.ucs_path) - 1);
    header.pid           = getpid();
    header.mode          = ucs_global_opts.profile_mode;
    header.num_locations = ucs_profile_global_ctx.num_locations;
    header.num_threads   = ucs_list_length(&ucs_profile_global_ctx.thread_list);
    header.one_second    = ucs_time_from_sec(1.0);
    ucs_profile_file_write_data(fd, &header, sizeof(header));

    /* write locations */
    status = ucs_profile_write_locations(fd);
    if (status != UCS_OK) {
        goto out_close_fd;
    }

    /* write threads */
    ucs_list_for_each(ctx, &ucs_profile_global_ctx.thread_list, list) {
        status = ucs_profile_file_write_thread(fd, ctx, write_time);
        if (status != UCS_OK) {
            goto out_close_fd;
        }
    }

out_close_fd:
    close(fd);
out_unlock:
    pthread_mutex_unlock(&ucs_profile_global_ctx.mutex);
}

static UCS_F_NOINLINE
ucs_profile_thread_context_t* ucs_profile_thread_init()
{
    ucs_profile_thread_context_t *ctx;
    size_t num_records;

    ucs_assert(ucs_global_opts.profile_mode);

    ctx = ucs_malloc(sizeof(*ctx), "profile_thread_context");
    if (ctx == NULL) {
        ucs_error("failed to allocate profiling thread context");
        return NULL;
    }

    ctx->tid        = ucs_get_tid();
    ctx->start_time = ucs_get_time();
    ctx->end_time   = 0;
    ctx->pthread_id = pthread_self();

    ucs_debug("profiling context %p: start on thread 0x%lx tid %d mode %d",
              ctx, (unsigned long)pthread_self(), ucs_get_tid(), 
              ucs_global_opts.profile_mode);

    /* Initialize log mode */
    if (ucs_global_opts.profile_mode & UCS_BIT(UCS_PROFILE_MODE_LOG)) {
        num_records = ucs_global_opts.profile_log_size /
                      sizeof(ucs_profile_record_t);
        ctx->log.start = ucs_calloc(num_records, sizeof(ucs_profile_record_t),
                                    "profile_log");
        if (ctx->log.start == NULL) {
            ucs_fatal("failed to allocate profiling log");
        }

        ctx->log.end        = ctx->log.start + num_records;
        ctx->log.current    = ctx->log.start;
        ctx->log.wraparound = 0;
    }

    /* Initialize accumulate mode */
    if (ucs_global_opts.profile_mode & UCS_BIT(UCS_PROFILE_MODE_ACCUM)) {
        ctx->accum.num_locations = 0;
        ctx->accum.locations     = NULL;
        ctx->accum.stack_top     = -1;
    }

    pthread_setspecific(ucs_profile_global_ctx.tls_key, ctx);

    pthread_mutex_lock(&ucs_profile_global_ctx.mutex);
    ucs_list_add_tail(&ucs_profile_global_ctx.thread_list, &ctx->list);
    pthread_mutex_unlock(&ucs_profile_global_ctx.mutex);

    return ctx;
}

static void ucs_profile_thread_cleanup(ucs_profile_thread_context_t *ctx)
{
    ucs_debug("profiling context %p: cleanup", ctx);

    if (ucs_global_opts.profile_mode & UCS_BIT(UCS_PROFILE_MODE_LOG)) {
        ucs_free(ctx->log.start);
    }

    if (ucs_global_opts.profile_mode & UCS_BIT(UCS_PROFILE_MODE_ACCUM)) {
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
 * @param [in]  type         Location type.
 * @param [in]  file         Source file name.
 * @param [in]  line         Source line number.
 * @param [in]  function     Calling function name.
 * @param [in]  name         Location name.
 * @param [out] loc_id_p     Filled with location ID:
 *                             0   - profiling is disabled
 *                             >0  - location index + 1
 */
static UCS_F_NOINLINE
int ucs_profile_get_location(ucs_profile_type_t type, const char *name,
                             const char *file, int line, const char *function,
                             volatile int *loc_id_p)
{
    ucs_profile_global_location_t *loc, *new_locations;
    int loc_id;

    pthread_mutex_lock(&ucs_profile_global_ctx.mutex);

    /* Check, with lock held, that the location is not already initialized */
    if (*loc_id_p >= 0) {
        loc_id = *loc_id_p;
        goto out_unlock;
    }

    /* Check if profiling is disabled */
    if (!ucs_global_opts.profile_mode) {
        *loc_id_p = loc_id = 0;
        goto out_unlock;
    }

    /* Location ID must be uninitialized */
    ucs_assert(*loc_id_p == -1);

    ucs_profile_for_each_location(loc) {
        if ((type == loc->super.type) && (line == loc->super.line) &&
            !strcmp(loc->super.name, name) &&
            !strcmp(loc->super.file, ucs_basename(file)) &&
            !strcmp(loc->super.function, function)) {
            goto out_found;
        }
    }

    ++ucs_profile_global_ctx.num_locations;

    /* Reallocate array if needed */
    if (ucs_profile_global_ctx.num_locations > ucs_profile_global_ctx.max_locations) {
        ucs_profile_global_ctx.max_locations =
                        2 * ucs_profile_global_ctx.num_locations;
        new_locations = ucs_realloc(ucs_profile_global_ctx.locations,
                                    sizeof(*ucs_profile_global_ctx.locations) *
                                    ucs_profile_global_ctx.max_locations,
                                    "profile_locations");
        if (new_locations == NULL) {
            ucs_warn("failed to expand locations array");
            *loc_id_p = loc_id = 0;
            goto out_unlock;
        }

        ucs_profile_global_ctx.locations = new_locations;
    }

    /* Initialize new location */
    loc = &ucs_profile_global_ctx.locations[ucs_profile_global_ctx.num_locations - 1];
    ucs_strncpy_zero(loc->super.file, ucs_basename(file), sizeof(loc->super.file));
    ucs_strncpy_zero(loc->super.function, function, sizeof(loc->super.function));
    ucs_strncpy_zero(loc->super.name, name, sizeof(loc->super.name));
    loc->super.line = line;
    loc->super.type = type;
    loc->loc_id_p   = loc_id_p;

out_found:
    *loc_id_p = loc_id = (loc - ucs_profile_global_ctx.locations) + 1;
    ucs_memory_cpu_store_fence();
out_unlock:
    pthread_mutex_unlock(&ucs_profile_global_ctx.mutex);
    return loc_id;
}

static void ucs_profile_thread_expand_locations(int loc_id)
{
    ucs_profile_thread_context_t *ctx;
    unsigned i, new_num_locations;

    ctx = pthread_getspecific(ucs_profile_global_ctx.tls_key);
    ucs_assert(ctx != NULL);

    new_num_locations = ucs_max(loc_id, ctx->accum.num_locations);
    ctx->accum.locations = ucs_realloc(ctx->accum.locations,
                                       sizeof(*ctx->accum.locations) *
                                       new_num_locations,
                                       "profile_thread_locations");
    if (ctx->accum.locations == NULL) {
        ucs_fatal("failed to allocate profiling per-thread locations");
    }

    for (i = ctx->accum.num_locations; i < new_num_locations; ++i) {
        ctx->accum.locations[i].count      = 0;
        ctx->accum.locations[i].total_time = 0;
    }

    ctx->accum.num_locations = new_num_locations;
}

void ucs_profile_record(ucs_profile_type_t type, const char *name,
                        uint32_t param32, uint64_t param64, const char *file,
                        int line, const char *function, volatile int *loc_id_p)
{
    ucs_profile_thread_location_t *loc;
    ucs_profile_thread_context_t *ctx;
    ucs_profile_record_t *rec;
    ucs_time_t current_time;
    int loc_id;

    /* If the location id is -1 or 0, need to re-read it with lock held */
    loc_id = *loc_id_p;
    if (ucs_unlikely(loc_id <= 0)) {
        loc_id = ucs_profile_get_location(type, name, file, line, function,
                                          loc_id_p);
        if (loc_id == 0) {
            return;
        }
    }

    ucs_memory_cpu_load_fence();

    ucs_assert(*loc_id_p                    != 0);
    ucs_assert(ucs_global_opts.profile_mode != 0);

    /* Get thread-specific profiling context */
    ctx = pthread_getspecific(ucs_profile_global_ctx.tls_key);
    if (ucs_unlikely(ctx == NULL)) {
        ctx = ucs_profile_thread_init();
    }

    current_time = ucs_get_time();
    if (ucs_global_opts.profile_mode & UCS_BIT(UCS_PROFILE_MODE_ACCUM)) {
        if (ucs_unlikely(loc_id > ctx->accum.num_locations)) {
            /* expand the locations array of the current thread */
            ucs_profile_thread_expand_locations(loc_id);
        }
        ucs_assert(loc_id - 1 < ctx->accum.num_locations);

        loc = &ctx->accum.locations[loc_id - 1];
        switch (type) {
        case UCS_PROFILE_TYPE_SCOPE_BEGIN:
            ctx->accum.stack[++ctx->accum.stack_top] = current_time;
            break;
        case UCS_PROFILE_TYPE_SCOPE_END:
            loc->total_time += current_time - ctx->accum.stack[ctx->accum.stack_top];
            --ctx->accum.stack_top;
            break;
        default:
            break;
        }
        ++loc->count;
    }

    if (ucs_global_opts.profile_mode & UCS_BIT(UCS_PROFILE_MODE_LOG)) {
        rec              = ctx->log.current;
        rec->timestamp   = current_time;
        rec->param64     = param64;
        rec->param32     = param32;
        rec->location    = loc_id - 1;
        if (++ctx->log.current >= ctx->log.end) {
            ctx->log.current    = ctx->log.start;
            ctx->log.wraparound = 1;
        }
    }
}

static void ucs_profile_check_active_threads()
{
    size_t num_active_threads;

    pthread_mutex_lock(&ucs_profile_global_ctx.mutex);
    num_active_threads = ucs_list_length(&ucs_profile_global_ctx.thread_list);
    pthread_mutex_unlock(&ucs_profile_global_ctx.mutex);

    if (num_active_threads > 0) {
        ucs_warn("%zd profiled threads are still running", num_active_threads);
    }
}

void ucs_profile_reset_locations()
{
    ucs_profile_global_location_t *loc;

    pthread_mutex_lock(&ucs_profile_global_ctx.mutex);

    ucs_profile_for_each_location(loc) {
        *loc->loc_id_p = -1;
    }

    ucs_profile_global_ctx.num_locations = 0;
    ucs_profile_global_ctx.max_locations = 0;
    ucs_free(ucs_profile_global_ctx.locations);
    ucs_profile_global_ctx.locations = NULL;

    pthread_mutex_unlock(&ucs_profile_global_ctx.mutex);
}

static void ucs_profile_cleanup_completed_threads()
{
    ucs_profile_thread_context_t *ctx, *tmp;

    pthread_mutex_lock(&ucs_profile_global_ctx.mutex);
    ucs_list_for_each_safe(ctx, tmp, &ucs_profile_global_ctx.thread_list,
                           list) {
        if (ctx->is_completed) {
            ucs_profile_thread_cleanup(ctx);
        }
    }
    pthread_mutex_unlock(&ucs_profile_global_ctx.mutex);
}

void ucs_profile_dump()
{
    ucs_profile_thread_context_t *ctx;

    /* finalize profiling on current thread */
    ctx = pthread_getspecific(ucs_profile_global_ctx.tls_key);
    if (ctx) {
        ucs_profile_thread_finalize(ctx);
        pthread_setspecific(ucs_profile_global_ctx.tls_key, NULL);
    }

    /* write and cleanup all completed threads (including the current thread) */
    ucs_profile_write();
    ucs_profile_cleanup_completed_threads();
}

void ucs_profile_global_init()
{
    if (ucs_global_opts.profile_mode && !strlen(ucs_global_opts.profile_file)) {
        // TODO make sure profiling file is writeable
        ucs_warn("profiling file not specified");
    }

    pthread_key_create(&ucs_profile_global_ctx.tls_key,
                       ucs_profile_thread_key_destr);
}

void ucs_profile_global_cleanup()
{
    ucs_profile_dump();
    ucs_profile_check_active_threads();
    pthread_key_delete(ucs_profile_global_ctx.tls_key);
}
