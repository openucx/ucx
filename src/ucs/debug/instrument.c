/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#include <ucs/debug/instrument.h>

#include <ucs/debug/log.h>
#include <ucs/sys/math.h>
#include <ucs/sys/sys.h>
#include <ucs/debug/debug.h>
#include <ucs/datastruct/sglib_wrapper.h>

#if HAVE_INSTRUMENTATION

ucs_instrument_context_t ucs_instr_ctx;

static void ucs_instrument_write_common(void *buffer, size_t size)
{
    ssize_t written = write(ucs_instr_ctx.fd, buffer, size);
    if (written < 0) {
        ucs_warn("failed to write %Zu bytes to instrumentation file: %m", size);
    } else if (size != written) {
        ucs_warn("wrote only %Zd of %Zu bytes to instrumentation file: %m",
                 written, size);
    }
}

static void ucs_instrument_write_location(ucs_instrument_location_t *location)
{
    ucs_instrument_write_common(location,
                                offsetof(ucs_instrument_location_t, list));
}

static void ucs_instrument_write_records(ucs_instrument_record_t *from,
                                         ucs_instrument_record_t *to)
{
    ucs_instrument_write_common(from, (char*)to - (char*)from);
}

static void ucs_instrument_fill_header(ucs_instrument_header_t *header)
{
    memset(header, 0, sizeof *header);

    /* Library */
    header->ucs_lib.base = ucs_debug_get_lib_base_addr();
    strncpy(header->ucs_lib.path, ucs_debug_get_lib_path(), sizeof(header->ucs_lib.path) - 1);
    if (strlen(header->ucs_lib.path)) {
        header->ucs_lib.chksum = ucs_file_checksum(header->ucs_lib.path);
    }

    /* Process */
    ucs_read_file(header->app.cmdline, sizeof(header->app.cmdline), 1, "/proc/self/cmdline");
    header->app.pid = getpid();
    strncpy(header->app.hostname, ucs_get_host_name(), sizeof(header->app.hostname) - 1);

    /* Samples */
    header->num_locations = ucs_list_length(&ucs_instr_ctx.locations_head);
    header->num_records   = ucs_min(ucs_instr_ctx.count - header->record_offset,
                                   ucs_instr_ctx.end - ucs_instr_ctx.start);
    header->record_offset = ucs_instr_ctx.count - header->num_records;
    header->start_time    = ucs_instr_ctx.start_time;
    header->one_second    = ucs_time_from_sec(1.0);
    header->one_record    = ucs_instr_ctx.one_record_time;
}

static void ucs_instrument_write()
{
    ucs_instrument_header_t header;

    /* write header */
    ucs_instrument_fill_header(&header);
    if (write(ucs_instr_ctx.fd, &header, sizeof(header)) < sizeof(header)) {
        ucs_warn("failed to write instrument header");
    }

    /* write header */
    while (!ucs_list_is_empty(&ucs_instr_ctx.locations_head)) {
        ucs_instrument_location_t *location_entry =
                ucs_list_extract_head(&ucs_instr_ctx.locations_head,
                        ucs_instrument_location_t, list);
        ucs_instrument_write_location(location_entry);
        ucs_free(location_entry);
    }

    /* write records */
    if (header.record_offset > 0) {
        ucs_instrument_write_records(ucs_instr_ctx.current, ucs_instr_ctx.end);
    }
    ucs_instrument_write_records(ucs_instr_ctx.start, ucs_instr_ctx.current);
}

void ucs_instrument_init()
{
    char fullpath[1024] = {0};
    char filename[1024] = {0};
    size_t num_records;

    if (strlen(ucs_global_opts.instrument_file) == 0) {
        goto disable;
    }

    ucs_fill_filename_template(ucs_global_opts.instrument_file,
                               filename, sizeof(filename));
    ucs_expand_path(filename, fullpath, sizeof(fullpath) - 1);

    ucs_instr_ctx.fd = open(fullpath, O_WRONLY|O_CREAT|O_TRUNC, 0600);
    if (ucs_instr_ctx.fd < 0) {
        ucs_warn("failed to open %s for writing: %m", fullpath);
        goto disable;
    }

    num_records = ucs_global_opts.instrument_max_size /
                  sizeof(ucs_instrument_record_t);
    ucs_instr_ctx.start = ucs_calloc(num_records,
                                     sizeof(ucs_instrument_record_t),
                                     "instrument_data_buffer");
    if (ucs_instr_ctx.start == NULL) {
        ucs_warn("failed to allocate instrumentation buffer");
        goto disable_close_file;
    }

    ucs_list_head_init(&ucs_instr_ctx.locations_head);
    ucs_instr_ctx.enabled          = ucs_global_opts.instrument_types;
    ucs_instr_ctx.end              = ucs_instr_ctx.start + num_records;
    ucs_instr_ctx.current          = ucs_instr_ctx.start;
    ucs_instr_ctx.count            = 0;
    ucs_instr_ctx.next_location_id = 1;

    /* Measure the time it takes to generate a single record */
    ucs_instr_ctx.start_time       = ucs_get_time();
    for (num_records = 0; num_records < 1000; num_records++) {
        ucs_instrument_record(0, 0, 0);
    }

    ucs_instr_ctx.current          = ucs_instr_ctx.start;
    ucs_instr_ctx.count            = 0;
    ucs_instr_ctx.one_record_time  = (ucs_instr_ctx.current->timestamp -
            ucs_instr_ctx.start_time) / num_records;

    ucs_info("saving instrumentation records to %s", fullpath);
    return;

disable_close_file:
    close(ucs_instr_ctx.fd);
disable:
    ucs_instr_ctx.fd = -1;
    ucs_instr_ctx.enabled = 0;
    ucs_trace("instrumentation is disabled");
}

void ucs_instrument_cleanup()
{
    if (ucs_instr_ctx.fd == -1) {
        return;
    }

    ucs_instrument_write();
    close(ucs_instr_ctx.fd);
    ucs_free(ucs_instr_ctx.start);
}

uint32_t ucs_instrument_register(ucs_instrumentation_types_t type,
                                 char const *type_name,
                                 char const *custom_name,
                                 char const *file_name,
                                 unsigned line_number)
{
    ucs_instrument_location_t *location_entry;
    ucs_instrument_context_t *ctx = &ucs_instr_ctx;

    if (((ctx->enabled & UCS_BIT(type)) == 0) &&
         (type != UCS_INSTRUMENT_TYPE_LAST)) { /* LAST used for measuring */
        return 0;
    }

    location_entry = ucs_calloc(1, sizeof(*location_entry),
                                "instrument_location");
    if (location_entry == NULL) {
        return 0;
    }

    /* Fill in a location entry */
    snprintf(location_entry->name, sizeof(location_entry->name) - 1,
             "%s - %s (%s:%i)", type_name, custom_name, file_name, line_number);
    ucs_list_add_tail(&ctx->locations_head, &location_entry->list);
    location_entry->location = ctx->next_location_id++;
    return location_entry->location;
}

void ucs_instrument_record(uint32_t location, uint64_t lparam, uint32_t wparam)
{
    ucs_instrument_context_t *ctx = &ucs_instr_ctx;
    ucs_instrument_record_t *current = ctx->current;

    current->timestamp = ucs_get_time();
    current->lparam    = lparam;
    current->wparam    = wparam;
    current->location  = location;

    ++ctx->count;
    ++ctx->current;
    if (ctx->current >= ctx->end) {
        ctx->current = ctx->start;
    }
}

#else

void ucs_instrument_init()
{
}

void ucs_instrument_cleanup()
{
}

#endif
