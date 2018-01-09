/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "profile.h"

#include <ucs/debug/log.h>
#include <ucs/sys/string.h>
#include <ucs/sys/sys.h>

#if HAVE_PROFILING

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

void ucs_profile_get_location(ucs_profile_type_t type, const char *name,
                              const char *file, int line, const char *function,
                              int *loc_id_p)
{
    ucs_profile_location_t *loc;
    int location;

    /* Check if profiling is disabled */
    if (!ucs_global_opts.profile_mode) {
        *loc_id_p = 0;
        return;
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
            return;
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
    *loc_id_p       = location + 1;
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

#else

void ucs_profile_global_init()
{
}

void ucs_profile_global_cleanup()
{
}

void ucs_profile_dump()
{
}

#endif
