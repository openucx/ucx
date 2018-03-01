/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "profile.h"

#include <ucs/debug/log.h>
#include <ucs/sys/string.h>
#include <ucs/sys/sys.h>
#include <ucs/time/time.h>
#include <pthread.h>


/**
 * Profiling global context
 */
typedef struct ucs_profile_global_context {

    ucs_profile_location_t   *locations;    /**< Array of all locations */
    unsigned                 num_locations; /**< Number of valid locations */
    unsigned                 max_locations; /**< Size of locations array */
    pthread_mutex_t          mutex;         /**< Protects updating the locations array */

    struct {
        ucs_profile_record_t *start, *end;  /**< Circular log buffer */
        ucs_profile_record_t *current;      /**< Current log pointer */
        int                  wraparound;    /**< Whether log was rotated */
    } log;

    struct {
        int                  stack_top;     /**< Index of stack top */
        ucs_time_t           stack[UCS_PROFILE_STACK_MAX]; /**< Timestamps for each nested scope */
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
    .num_locations   = 0,
    .max_locations   = 0,
    .mutex           = PTHREAD_MUTEX_INITIALIZER
};

static void ucs_profile_file_write_data(int fd, void *data, size_t size)
{
    ssize_t written = write(fd, data, size);
    if (written < 0) {
        ucs_warn("failed to write %zu bytes to profiling file: %m", size);
    } else if (size != written) {
        ucs_warn("wrote only %zd of %zu bytes to profiling file: %m",
                 written, size);
    }
}

static void ucs_profile_file_write_records(int fd, ucs_profile_record_t *begin,
                                           ucs_profile_record_t *end)
{
    ucs_profile_file_write_data(fd, begin, (void*)end - (void*)begin);
}

static void ucs_profile_write()
{
    ucs_profile_header_t header;
    char fullpath[1024] = {0};
    char filename[1024] = {0};
    int fd;

    if (!ucs_global_opts.profile_mode) {
        return;
    }

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
    header.pid = getpid();
    header.mode = ucs_global_opts.profile_mode;
    header.num_locations = ucs_profile_ctx.num_locations;
    header.num_records   = ucs_profile_ctx.log.wraparound ?
                    (ucs_profile_ctx.log.end     - ucs_profile_ctx.log.start) :
                    (ucs_profile_ctx.log.current - ucs_profile_ctx.log.start);
    header.one_second    = ucs_time_from_sec(1.0);
    ucs_profile_file_write_data(fd, &header, sizeof(header));

    /* write locations */
    ucs_profile_file_write_data(fd, ucs_profile_ctx.locations,
                                sizeof(*ucs_profile_ctx.locations) *
                                ucs_profile_ctx.num_locations);

    /* write records */
    if (ucs_profile_ctx.log.wraparound > 0) {
        ucs_profile_file_write_records(fd, ucs_profile_ctx.log.current,
                                       ucs_profile_ctx.log.end);
    }
    ucs_profile_file_write_records(fd, ucs_profile_ctx.log.start,
                                   ucs_profile_ctx.log.current);

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
    ucs_profile_location_t *loc;
    int location;

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
    }

    /* Initialize new location */
    loc             = &ucs_profile_ctx.locations[location];
    ucs_strncpy_zero(loc->file, basename(file), sizeof(loc->file));
    ucs_strncpy_zero(loc->function, function, sizeof(loc->function));
    ucs_strncpy_zero(loc->name, name, sizeof(loc->name));
    loc->line       = line;
    loc->type       = type;
    loc->total_time = 0;
    loc->count      = 0;
    loc->loc_id_p   = loc_id_p;

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
    ucs_profile_record_t   *rec;
    ucs_profile_location_t *loc;
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
        loc = &ctx->locations[loc_id - 1];
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

    ucs_info("profiling is enabled");
    return;

disable:
    ucs_global_opts.profile_mode = 0;
off:
    ucs_trace("profiling is disabled");
}

static void ucs_profile_reset_locations()
{
    ucs_profile_location_t *loc;

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
    ucs_profile_reset_locations();
}

void ucs_profile_dump()
{
    ucs_profile_location_t *loc;

    ucs_profile_write();

    for (loc = ucs_profile_ctx.locations;
         loc < ucs_profile_ctx.locations + ucs_profile_ctx.num_locations;
         ++loc)
    {
        loc->count      = 0;
        loc->total_time = 0;
    }

    if (ucs_global_opts.profile_mode & UCS_BIT(UCS_PROFILE_MODE_LOG)) {
        ucs_profile_ctx.log.wraparound = 0;
        ucs_profile_ctx.log.current    = ucs_profile_ctx.log.start;
    }
}
