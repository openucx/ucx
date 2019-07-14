/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "profile.h"

#include <ucs/debug/debug.h>
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
    ucs_time_t                    start_time;    /**< Thread context init time */
    ucs_time_t                    end_time;      /**< Thread end time */

    struct {
        ucs_profile_record_t      *start, *end;  /**< Circular log buffer */
        ucs_profile_record_t      *current;      /**< Current log pointer */
        int                       wraparound;    /**< Whether log was rotated */
    } log;

    struct {
        int                       stack_top;     /**< Index of stack top */
        ucs_time_t                stack[UCS_PROFILE_STACK_MAX]; /**< Timestamps for each nested scope */
        ucs_profile_thread_location_t *thread_locations; /**< Statistics per location */
   } accum;

} ucs_profile_global_context_t;


const char *ucs_profile_mode_names[] = {
    [UCS_PROFILE_MODE_ACCUM] = "accum",
    [UCS_PROFILE_MODE_LOG]   = "log",
    [UCS_PROFILE_MODE_LAST]  = NULL
};

ucs_profile_global_context_t ucs_profile_ctx = {
    .locations       = NULL,
    .log.start       = NULL,
    .log.end         = NULL,
    .log.current     = NULL,
    .log.wraparound  = 0,
    .accum.stack_top = -1,
    .accum.thread_locations = NULL,
    .num_locations   = 0,
    .max_locations   = 0,
    .mutex           = PTHREAD_MUTEX_INITIALIZER
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
    return ucs_profile_file_write_data(fd, begin, (void*)end - (void*)begin);
}

static ucs_status_t
ucs_profile_file_write_thread(int fd, ucs_time_t default_end_time)
{
    ucs_profile_thread_location_t empty_location = { .total_time = 0, .count = 0 };
    ucs_profile_thread_header_t thread_hdr;
    unsigned i, num_locations;
    ucs_status_t status;

    /* write thread header */
    memset(&thread_hdr, 0, sizeof(thread_hdr));
    thread_hdr.tid        = getpid();
    thread_hdr.start_time = ucs_profile_ctx.start_time;
    thread_hdr.end_time   = ucs_profile_ctx.end_time;

    if (ucs_global_opts.profile_mode & UCS_BIT(UCS_PROFILE_MODE_LOG)) {
        thread_hdr.num_records = ucs_profile_ctx.log.wraparound ?
                        (ucs_profile_ctx.log.end     - ucs_profile_ctx.log.start) :
                        (ucs_profile_ctx.log.current - ucs_profile_ctx.log.start);
    } else {
        thread_hdr.num_records = 0;
    }

    status = ucs_profile_file_write_data(fd, &thread_hdr, sizeof(thread_hdr));
    if (status != UCS_OK) {
        return status;
    }

    /* If accumulate mode is not enabled, there are no location entries */
    if (ucs_global_opts.profile_mode & UCS_BIT(UCS_PROFILE_MODE_ACCUM)) {
        num_locations = ucs_profile_ctx.num_locations;
    } else {
        num_locations = 0;
    }

    /* write profiling information for every location
     * note: the thread location array may be smaller (or even empty) than the
     * global list, but it cannot be larger. If it's smaller, we pad with empty
     * entries
     */
    ucs_profile_file_write_data(fd, ucs_profile_ctx.accum.thread_locations,
                                num_locations *
                                sizeof(*ucs_profile_ctx.accum.thread_locations));
    for (i = num_locations; i < ucs_profile_ctx.num_locations; ++i) {
        status = ucs_profile_file_write_data(fd, &empty_location,
                                             sizeof(empty_location));
        if (status != UCS_OK) {
            return status;
        }
    }

    /* write profiling records */
    if (ucs_global_opts.profile_mode & UCS_BIT(UCS_PROFILE_MODE_LOG)) {
        if (ucs_profile_ctx.log.wraparound) {
            status = ucs_profile_file_write_records(fd,
                                                    ucs_profile_ctx.log.current,
                                                    ucs_profile_ctx.log.end);
            if (status != UCS_OK) {
                return status;
            }
        }

        status = ucs_profile_file_write_records(fd, ucs_profile_ctx.log.start,
                                                ucs_profile_ctx.log.current);
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

    for (loc = ucs_profile_ctx.locations;
         loc < ucs_profile_ctx.locations + ucs_profile_ctx.num_locations;
         ++loc)
    {
        status = ucs_profile_file_write_data(fd, &loc->super, sizeof(loc->super));
        if (status != UCS_OK) {
            return status;
        }
    }

    return UCS_OK;
}

static void ucs_profile_write()
{
    ucs_profile_header_t header;
    char fullpath[1024] = {0};
    char filename[1024] = {0};
    ucs_time_t write_time;
    ucs_status_t status;
    int fd;

    if (!ucs_global_opts.profile_mode) {
        return;
    }

    write_time = ucs_get_time();

    ucs_fill_filename_template(ucs_global_opts.profile_file,
                               filename, sizeof(filename));
    ucs_expand_path(filename, fullpath, sizeof(fullpath) - 1);

    fd = open(fullpath, O_WRONLY|O_CREAT|O_TRUNC, 0600);
    if (fd < 0) {
        ucs_error("failed to write profiling data to '%s': %m", fullpath);
        return;
    }

    /* write header */
    memset(&header, 0, sizeof(header));
    ucs_read_file(header.cmdline, sizeof(header.cmdline), 1, "/proc/self/cmdline");
    strncpy(header.hostname, ucs_get_host_name(), sizeof(header.hostname) - 1);
    header.version       = UCS_PROFILE_FILE_VERSION;
    strncpy(header.ucs_path, ucs_debug_get_lib_path(), sizeof(header.ucs_path) - 1);
    header.pid           = getpid();
    header.mode          = ucs_global_opts.profile_mode;
    header.num_locations = ucs_profile_ctx.num_locations;
    header.num_threads   = 1;
    header.one_second    = ucs_time_from_sec(1.0);
    ucs_profile_file_write_data(fd, &header, sizeof(header));

    /* write locations */
    status = ucs_profile_write_locations(fd);
    if (status != UCS_OK) {
        goto out_close_fd;
    }

    /* write threads */
    status = ucs_profile_file_write_thread(fd, write_time);
    if (status != UCS_OK) {
        goto out_close_fd;
    }

out_close_fd:
    close(fd);
}

/*
 * Register a profiling location - should be called once per location in the
 * code, before the first record of each such location is made.
 * SHOULD NOT be used directly - use UCS_PROFILE macros instead.
 *
 * @param [in]  type      Location type.
 * @param [in]  file      Source file name.
 * @param [in]  line      Source line number.
 * @param [in]  function  Calling function name.
 * @param [in]  name      Location name.
 * @param [out] loc_id_p  Filled with location ID:
 *                          0   - profiling is disabled
 *                          >0  - location index + 1
 */
static void ucs_profile_get_location(ucs_profile_type_t type, const char *name,
                                     const char *file, int line,
                                     const char *function, volatile int *loc_id_p)
{
    ucs_profile_global_context_t *ctx = &ucs_profile_ctx;
    ucs_profile_global_location_t *loc;
    int location;
    int i;

    pthread_mutex_lock(&ucs_profile_ctx.mutex);

    if (*loc_id_p == 0) {
        goto out_unlock;
    }

    /* Check if profiling is disabled */
    if (!ucs_global_opts.profile_mode) {
        *loc_id_p = 0;
        goto out_unlock;
    }

    /* Location ID must be uninitialized */
    ucs_assert(*loc_id_p == -1);

    for (i = 0; i < ctx->num_locations; ++i) {
        loc = &ctx->locations[i];

        if ((type == loc->super.type) &&
            (line == loc->super.line) &&
            !strcmp(loc->super.name, name) &&
            !strcmp(loc->super.file, basename(file)) &&
            !strcmp(loc->super.function, function)) {

            *loc_id_p = i + 1;
            goto out_unlock;
        }
    }

    location = ucs_profile_ctx.num_locations++;

    /* Reallocate array if needed */
    if (ucs_profile_ctx.num_locations > ucs_profile_ctx.max_locations) {
        ucs_profile_ctx.max_locations = ucs_profile_ctx.num_locations * 2;
        ucs_profile_ctx.locations = ucs_realloc(ucs_profile_ctx.locations,
                                                sizeof(*ucs_profile_ctx.locations) *
                                                ucs_profile_ctx.max_locations,
                                                "profile_locations");
        if (ucs_profile_ctx.locations == NULL) {
            ucs_warn("failed to expand locations array");
            *loc_id_p = 0;
            goto out_unlock;
        }

        if (ucs_global_opts.profile_mode & UCS_BIT(UCS_PROFILE_MODE_ACCUM)) {
            ucs_profile_ctx.accum.thread_locations =
                            ucs_realloc(ucs_profile_ctx.accum.thread_locations,
                                        sizeof(*ucs_profile_ctx.accum.thread_locations) *
                                        ucs_profile_ctx.max_locations,
                                        "profile_thread_locations");
            if (ucs_profile_ctx.accum.thread_locations == NULL) {
                ucs_warn("failed to expand thread locations array");
                *loc_id_p = 0;
                goto out_unlock;
            }
        }
    }

    /* Initialize new location */
    loc             = &ucs_profile_ctx.locations[location];
    ucs_strncpy_zero(loc->super.file, basename(file), sizeof(loc->super.file));
    ucs_strncpy_zero(loc->super.function, function, sizeof(loc->super.function));
    ucs_strncpy_zero(loc->super.name, name, sizeof(loc->super.name));
    loc->super.line = line;
    loc->super.type = type;
    loc->loc_id_p   = loc_id_p;

    if (ucs_global_opts.profile_mode & UCS_BIT(UCS_PROFILE_MODE_ACCUM)) {
        ucs_profile_ctx.accum.thread_locations[location].total_time = 0;
        ucs_profile_ctx.accum.thread_locations[location].count      = 0;
    }

    ucs_memory_cpu_store_fence();
    *loc_id_p       = location + 1;

out_unlock:
    pthread_mutex_unlock(&ucs_profile_ctx.mutex);
}

void ucs_profile_record(ucs_profile_type_t type, const char *name,
                        uint32_t param32, uint64_t param64, const char *file,
                        int line, const char *function, volatile int *loc_id_p)
{
    extern ucs_profile_global_context_t ucs_profile_ctx;
    ucs_profile_global_context_t *ctx = &ucs_profile_ctx;
    ucs_profile_record_t          *rec;
    ucs_profile_thread_location_t *loc;
    ucs_time_t current_time;
    int loc_id;

    /* If the location id is -1 or 0, need to re-read it with lock held */
    if (ucs_unlikely((loc_id = *loc_id_p) <= 0)) {
        ucs_profile_get_location(type, name, file, line, function, loc_id_p);
        if ((loc_id = *loc_id_p) == 0) {
            return;
        }
    }

    ucs_memory_cpu_load_fence();
    ucs_assert(*loc_id_p                    != 0);
    ucs_assert(ucs_global_opts.profile_mode != 0);

    current_time = ucs_get_time();
    if (ucs_global_opts.profile_mode & UCS_BIT(UCS_PROFILE_MODE_ACCUM)) {
        loc = &ctx->accum.thread_locations[loc_id - 1];
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


void ucs_profile_global_init()
{
    size_t num_records;

    if (!ucs_global_opts.profile_mode) {
        goto off;
    }

    if (!strlen(ucs_global_opts.profile_file)) {
        ucs_warn("profiling file not specified, profiling is disabled");
        goto disable;
    }

    if (ucs_global_opts.profile_mode & UCS_BIT(UCS_PROFILE_MODE_LOG)) {
        num_records = ucs_global_opts.profile_log_size / sizeof(ucs_profile_record_t);
        ucs_profile_ctx.log.start = ucs_calloc(num_records,
                                               sizeof(ucs_profile_record_t),
                                               "profile_log");
        if (ucs_profile_ctx.log.start == NULL) {
            ucs_warn("failed to allocate profiling log");
            goto disable;
        }

        ucs_profile_ctx.log.end     = ucs_profile_ctx.log.start + num_records;
        ucs_profile_ctx.log.current = ucs_profile_ctx.log.start;
    }

    if (ucs_global_opts.profile_mode & UCS_BIT(UCS_PROFILE_MODE_ACCUM)) {
        ucs_profile_ctx.accum.stack_top = -1;
    }

    ucs_debug("profiling is enabled");
    return;

disable:
    ucs_global_opts.profile_mode = 0;
off:
    ucs_trace("profiling is disabled");
}

void ucs_profile_reset_locations()
{
    ucs_profile_global_location_t *loc;

    pthread_mutex_lock(&ucs_profile_ctx.mutex);
    for (loc = ucs_profile_ctx.locations;
         loc < ucs_profile_ctx.locations + ucs_profile_ctx.num_locations;
         ++loc)
    {
        *loc->loc_id_p = -1;
    }

    ucs_profile_ctx.num_locations = 0;
    ucs_profile_ctx.max_locations = 0;
    ucs_free(ucs_profile_ctx.locations);
    ucs_profile_ctx.locations = NULL;
    pthread_mutex_unlock(&ucs_profile_ctx.mutex);
}

void ucs_profile_global_cleanup()
{
    ucs_profile_write();
    ucs_free(ucs_profile_ctx.log.start);
    ucs_profile_ctx.log.start      = NULL;
    ucs_profile_ctx.log.end        = NULL;
    ucs_profile_ctx.log.current    = NULL;
    ucs_profile_ctx.log.wraparound = 0;
}

void ucs_profile_dump()
{
    ucs_profile_thread_location_t *loc;

    ucs_profile_write();

    if (ucs_global_opts.profile_mode & UCS_BIT(UCS_PROFILE_MODE_ACCUM)) {
        for (loc = ucs_profile_ctx.accum.thread_locations;
             loc < ucs_profile_ctx.accum.thread_locations + ucs_profile_ctx.num_locations;
             ++loc)
        {
            loc->count      = 0;
            loc->total_time = 0;
        }
    }

    if (ucs_global_opts.profile_mode & UCS_BIT(UCS_PROFILE_MODE_LOG)) {
        ucs_profile_ctx.log.wraparound = 0;
        ucs_profile_ctx.log.current    = ucs_profile_ctx.log.start;
    }
}
