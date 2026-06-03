/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#include "ucs/type/status.h"
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "table.h"

#include <ucs/debug/assert.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/math.h>
#include <ucs/sys/string.h>
#include <string.h>
#include <ucs/debug/log.h>

ucs_status_t
ucs_table_init(ucs_table_t *table, const ucs_table_config_t *config)
{
    if (config->n_cols <= 0) {
        ucs_error("number of columns must be positive (n_cols: %u)",
                  config->n_cols);
        return UCS_ERR_INVALID_PARAM;
    }

    table->config = *config;
    ucs_array_init_dynamic(&table->entries);

    return UCS_OK;
}

void ucs_table_cleanup(ucs_table_t *table)
{
    ucs_table_entry_t *entry;
    ucs_table_cell_t *cell;

    ucs_array_for_each(entry, &table->entries) {
        if (entry->kind != UCS_TABLE_ENTRY_ROW) {
            continue;
        }

        ucs_array_for_each(cell, &entry->cells) {
            ucs_string_buffer_cleanup(&cell->text);
        }

        ucs_array_cleanup_dynamic(&entry->cells);
    }

    ucs_array_cleanup_dynamic(&table->entries);
}

ucs_status_t ucs_table_add_separator(ucs_table_t *table)
{
    ucs_table_entry_t *entry;

    entry = ucs_array_append(&table->entries,
                             ucs_error("failed to add table separator entry"));
    if (entry == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    entry->kind = UCS_TABLE_ENTRY_SEPARATOR;
    return UCS_OK;
}

ucs_status_t ucs_table_add_row(ucs_table_t *table, ucs_table_row_h *row_p)
{
    ucs_table_entry_t *entry;
    ucs_status_t status;

    entry = ucs_array_append(&table->entries,
                             ucs_error("failed to add table row entry"));
    if (entry == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    entry->kind = UCS_TABLE_ENTRY_ROW;
    ucs_array_init_dynamic(&entry->cells);

    /* Pre-reserve so cell pointers stay stable across add_cell. */
    status = ucs_array_reserve(&entry->cells, table->config.n_cols);
    if (status != UCS_OK) {
        ucs_error("failed to reserve table row cells");
        return status;
    }

    *row_p = ucs_array_length(&table->entries) - 1;
    return UCS_OK;
}

static ucs_table_cells_t *
ucs_table_row_cells(ucs_table_t *table, ucs_table_row_h row)
{
    ucs_table_entry_t *entry = &ucs_array_elem(&table->entries, row);

    ucs_assert(entry->kind == UCS_TABLE_ENTRY_ROW);
    return &entry->cells;
}

static ucs_status_t
ucs_table_row_add_cell(ucs_table_t *table, ucs_table_row_h row,
                       unsigned col_span, ucs_table_align_t align,
                       ucs_table_cell_t **cell_p)
{
    ucs_table_cells_t *cells = ucs_table_row_cells(table, row);
    ucs_table_cell_t *cell;

    if (col_span <= 0) {
        ucs_error("table column span must be positive (col_span: %u)",
                  col_span);
        return UCS_ERR_INVALID_PARAM;
    }

    cell = ucs_array_append(cells, ucs_error("failed to add table cell"));
    if (cell == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    cell->col_span = col_span;
    cell->align    = align;
    ucs_string_buffer_init(&cell->text);

    if (cell_p != NULL) {
        *cell_p = cell;
    }

    return UCS_OK;
}

ucs_status_t ucs_table_row_add_cell_empty(ucs_table_t *table,
                                          ucs_table_row_h row,
                                          unsigned col_span)
{
    return ucs_table_row_add_cell(table, row, col_span, UCS_TABLE_ALIGN_LEFT,
                                  NULL);
}

ucs_status_t ucs_table_row_add_cell_fmt(ucs_table_t *table, ucs_table_row_h row,
                                        unsigned col_span,
                                        ucs_table_align_t align,
                                        const char *fmt, ...)
{
    ucs_table_cell_t *cell;
    ucs_status_t status;
    const char *cstr;
    va_list ap;

    status = ucs_table_row_add_cell(table, row, col_span, align, &cell);
    if (status != UCS_OK) {
        return status;
    }

    va_start(ap, fmt);
    ucs_string_buffer_vappendf(&cell->text, fmt, ap);
    va_end(ap);

    cstr = ucs_string_buffer_cstr(&cell->text);
    if (strchr(cstr, '\n') != NULL) {
        ucs_error("table cell content must not contain newline: '%s'", cstr);
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

/* Calculate the total visible width of a cell spanning `col_span` columns. */
static unsigned
ucs_table_cell_character_width(const ucs_table_t *table, const unsigned *widths,
                               unsigned start, unsigned col_span)
{
    unsigned width = 0;
    unsigned i;

    ucs_assertv(col_span > 0, "column span must be positive (col_span: %u)",
                col_span);
    ucs_assertv(
            start + col_span <= table->config.n_cols,
            "column span exceeds number of columns (start: %u, col_span: %u)",
            start, col_span);

    for (i = 0; i < col_span; ++i) {
        width += widths[start + i];
    }

    width += 3 * (col_span - 1);
    return width;
}

static ucs_status_t
ucs_table_compute_widths(const ucs_table_t *table, unsigned *widths)
{
    ucs_table_entry_t *entry;
    ucs_table_cell_t *cell;
    unsigned i, col, content_len;
    unsigned existing;

    for (i = 0; i < table->config.n_cols; ++i) {
        widths[i] = 0;
    }

    /* Pass 1: col_span == 1 cells. */
    ucs_array_for_each(entry, &table->entries) {
        if (entry->kind != UCS_TABLE_ENTRY_ROW) {
            continue;
        }

        col = 0;
        ucs_array_for_each(cell, &entry->cells) {
            if (col + cell->col_span > table->config.n_cols) {
                ucs_error("table row column span exceeds number of columns "
                          "(col: %u, col_span: %u, n_cols: %u)",
                          col, cell->col_span, table->config.n_cols);
                return UCS_ERR_INVALID_PARAM;
            }

            if (cell->col_span == 1) {
                content_len = ucs_string_buffer_length(&cell->text);
                widths[col] = ucs_max(widths[col], content_len);
            }

            col += cell->col_span;
        }
    }

    /* Pass 2: merged cells expand the rightmost spanned column. */
    ucs_array_for_each(entry, &table->entries) {
        if (entry->kind != UCS_TABLE_ENTRY_ROW) {
            continue;
        }

        col = 0;
        ucs_array_for_each(cell, &entry->cells) {
            if (cell->col_span > 1) {
                content_len = ucs_string_buffer_length(&cell->text);
                existing = ucs_table_cell_character_width(table, widths, col,
                                                          cell->col_span);

                if (content_len > existing) {
                    widths[col + cell->col_span - 1] += content_len - existing;
                }
            }

            col += cell->col_span;
        }
    }

    /* Equal-widths: widen every column to the max */
    if (table->config.equal_widths) {
        unsigned max_width = 0;
        for (i = 0; i < table->config.n_cols; ++i) {
            max_width = ucs_max(max_width, widths[i]);
        }

        for (i = 0; i < table->config.n_cols; ++i) {
            widths[i] = max_width;
        }
    }

    return UCS_OK;
}

/* Format a single cell at the given pixel width, branching on alignment. */
static void ucs_table_render_cell(ucs_string_buffer_t *strb,
                                  const ucs_table_cell_t *cell,
                                  unsigned pixel_width)
{
    const char *cstr = ucs_string_buffer_cstr(&cell->text);
    int content_len, pad, left_pad, right_pad;

    switch (cell->align) {
    case UCS_TABLE_ALIGN_LEFT:
        ucs_string_buffer_appendf(strb, "| %-*s ", (int)pixel_width, cstr);
        break;

    case UCS_TABLE_ALIGN_RIGHT:
        ucs_string_buffer_appendf(strb, "| %*s ", (int)pixel_width, cstr);
        break;

    case UCS_TABLE_ALIGN_CENTER:
        content_len = (int)strlen(cstr);
        pad         = ucs_max((int)pixel_width - content_len, 0);
        left_pad    = pad / 2;
        right_pad   = pad - left_pad;
        ucs_string_buffer_appendf(strb, "| %*s%s%*s ", left_pad, "", cstr,
                                  right_pad, "");
        break;

    default:
        ucs_fatal("invalid cell alignment %d", cell->align);
    }
}

static void
ucs_table_append_row_prefix(const ucs_table_t *table, ucs_string_buffer_t *strb)
{
    if (table->config.row_prefix != NULL) {
        ucs_string_buffer_appendf(strb, "%s", table->config.row_prefix);
    }
}

/* Render one row. The closing "|" has no trailing newline. */
static void ucs_table_render_cells(const ucs_table_t *table,
                                   const unsigned *widths,
                                   const ucs_table_cells_t *cells,
                                   ucs_string_buffer_t *strb)
{
    const ucs_table_cell_t *cell;
    unsigned col = 0;

    ucs_table_append_row_prefix(table, strb);

    ucs_array_for_each(cell, cells) {
        ucs_table_render_cell(strb, cell,
                              ucs_table_cell_character_width(table, widths, col,
                                                             cell->col_span));
        col += cell->col_span;
    }
    ucs_string_buffer_appendf(strb, "|");
}

static void ucs_table_render_separator(const ucs_table_t *table,
                                       const unsigned *widths,
                                       ucs_string_buffer_t *strb)
{
    unsigned i;

    ucs_table_append_row_prefix(table, strb);

    for (i = 0; i < table->config.n_cols; ++i) {
        ucs_string_buffer_appendc(strb, '+', 1);
        ucs_string_buffer_appendc(strb, '-', widths[i] + 2);
    }

    ucs_string_buffer_appendc(strb, '+', 1);
    ucs_string_buffer_appendc(strb, '\n', 1);
}

ucs_status_t
ucs_table_render(ucs_table_t const *table, ucs_string_buffer_t *strb)
{
    unsigned *widths = ucs_alloca(table->config.n_cols * sizeof(*widths));
    const ucs_table_entry_t *entry;
    ucs_status_t status;
    unsigned i;

    status = ucs_table_compute_widths(table, widths);
    if (status != UCS_OK) {
        return status;
    }

    /* Top frame */
    ucs_table_render_separator(table, widths, strb);

    /* Body rows and separators */
    for (i = 0; i < ucs_array_length(&table->entries); ++i) {
        entry = &ucs_array_elem(&table->entries, i);
        switch (entry->kind) {
        case UCS_TABLE_ENTRY_ROW:
            ucs_table_render_cells(table, widths, &entry->cells, strb);
            ucs_string_buffer_appendf(strb, "\n");
            break;

        case UCS_TABLE_ENTRY_SEPARATOR:
            ucs_table_render_separator(table, widths, strb);
            break;

        default:
            ucs_fatal("invalid table entry kind %d", entry->kind);
        }
    }

    /* Bottom frame; skip when the last entry is already a separator. */
    if (ucs_array_is_empty(&table->entries) ||
        (ucs_array_last(&table->entries)->kind != UCS_TABLE_ENTRY_SEPARATOR)) {
        ucs_table_render_separator(table, widths, strb);
    }

    return UCS_OK;
}

ucs_status_t ucs_table_print(ucs_table_t const *table)
{
    ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;
    ucs_status_t status;

    status = ucs_table_render(table, &strb);
    if (status != UCS_OK) {
        goto out;
    }

    printf("%s", ucs_string_buffer_cstr(&strb));

out:
    ucs_string_buffer_cleanup(&strb);
    return status;
}
