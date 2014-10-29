/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2011.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/


#include <ucs/debug/instrument.h>

#include <ucs/debug/log.h>
#include <ucs/sys/math.h>
#include <ucs/sys/sys.h>
#include <ucs/debug/debug.h>

#if HAVE_INSTRUMENTATION

ucs_instrument_context_t ucs_instr_ctx;

static void ucs_instrument_write_records(ucs_instrument_record_t *from,
                                         ucs_instrument_record_t *to)
{
    ssize_t written;
    size_t size;

    size = (char*)to - (char*)from;
    written = write(ucs_instr_ctx.fd, (void*)from, size);
    if (written < 0) {
        ucs_warn("failed to write %Zu bytes to instrumentation file: %m", size);
    } else if (size != written) {
        ucs_warn("wrote only %Zd of %Zu bytes to instrumentation file: %m", written, size);
    }
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
    header->num_records   = ucs_min(ucs_instr_ctx.count - header->record_offset,
                                   ucs_instr_ctx.end - ucs_instr_ctx.start);
    header->record_offset = ucs_instr_ctx.count - header->num_records;
    header->start_time    = ucs_instr_ctx.start_time;
    header->one_second    = ucs_time_from_sec(1.0);
}

static void ucs_instrument_write()
{
    ucs_instrument_header_t header;

    /* write header */
    ucs_instrument_fill_header(&header);
    if (write(ucs_instr_ctx.fd, &header, sizeof(header)) < sizeof(header)) {
        ucs_warn("failed to write instrument header");
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

    ucs_fill_filename_template(ucs_global_opts.instrument_file, filename, sizeof(filename));
    ucs_expand_path(filename, fullpath, sizeof(fullpath) - 1);

    ucs_instr_ctx.fd = open(fullpath, O_WRONLY|O_CREAT|O_TRUNC, 0600);
    if (ucs_instr_ctx.fd < 0) {
        ucs_warn("failed to open %s for writing: %m", fullpath);
        goto disable;
    }

    num_records = ucs_global_opts.instrument_max_size / sizeof(ucs_instrument_record_t);
    ucs_instr_ctx.start = calloc(num_records, sizeof(ucs_instrument_record_t));
    if (ucs_instr_ctx.start == NULL) {
        ucs_warn("failed to allocate instrumentation buffer");
        goto disable_close_file;
    }

    ucs_instr_ctx.enable     = 1;
    ucs_instr_ctx.end        = ucs_instr_ctx.start + num_records;
    ucs_instr_ctx.current    = ucs_instr_ctx.start;
    ucs_instr_ctx.count      = 0;
    ucs_instr_ctx.start_time = ucs_get_time();

    ucs_info("saving instrumentation records to %s", fullpath);
    return;

disable_close_file:
    close(ucs_instr_ctx.fd);
disable:
    ucs_instr_ctx.enable = 0;
    ucs_trace("instrumentation is disabled");
}

void ucs_instrument_cleanup()
{
    if (!ucs_instr_ctx.enable) {
        return;
    }

    ucs_instrument_write();
    close(ucs_instr_ctx.fd);
    free(ucs_instr_ctx.start);
}

void __ucs_instrument_record(uint64_t location, uint64_t lparam, uint32_t wparam)
{
    ucs_instrument_context_t *ctx = &ucs_instr_ctx;
    ucs_instrument_record_t *current = ctx->current;

    current->timestamp = ucs_get_time();
    current->lparam    = lparam;
    current->wparam    = wparam;
    current->location  = (uint32_t)location; /* chop off high dword */

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
