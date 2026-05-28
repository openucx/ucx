/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "table.h"

#include <ucs/datastruct/array.h>
#include <ucs/debug/assert.h>
#include <ucs/debug/log_def.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/math.h>
#include <ucs/sys/string.h>
#include <stdio.h>
#include <string.h>


void ucs_table_init(ucs_table_t *table, const ucs_table_config_t *config)
{
    table->config = *config;
    ucs_array_init_dynamic(&table->entries);

    ucs_assertv(config->n_cols > 0,
                "number of columns must be positive (n_cols: %u)",
                config->n_cols);
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


void ucs_table_add_separator(ucs_table_t *table)
{
    ucs_table_entry_t *entry;

    entry       = ucs_array_append_fixed(&table->entries);
    entry->kind = UCS_TABLE_ENTRY_SEPARATOR;
}


ucs_table_row_h ucs_table_add_row(ucs_table_t *table)
{
    ucs_table_entry_t *entry;
    ucs_status_t status;

    entry       = ucs_array_append_fixed(&table->entries);
    entry->kind = UCS_TABLE_ENTRY_ROW;
    ucs_array_init_dynamic(&entry->cells);

    /* Pre-reserve so cell pointers stay stable across add_cell. */
    status = ucs_array_reserve(&entry->cells, table->config.n_cols);
    if (status != UCS_OK) {
        ucs_fatal("failed to reserve table row cells");
    }

    return ucs_array_length(&table->entries) - 1;
}


static ucs_table_cells_t *
ucs_table_row_cells(ucs_table_t *table, ucs_table_row_h row)
{
    ucs_table_entry_t *entry = &ucs_array_elem(&table->entries, row);

    ucs_assert(entry->kind == UCS_TABLE_ENTRY_ROW);
    return &entry->cells;
}


static ucs_table_cell_t *
ucs_table_row_add_cell(ucs_table_t *table, ucs_table_row_h row,
                       unsigned col_span, ucs_table_align_t align)
{
    ucs_table_cells_t *cells = ucs_table_row_cells(table, row);
    ucs_table_cell_t *cell   = ucs_array_append_fixed(cells);

    cell->col_span = col_span;
    cell->align    = align;
    ucs_string_buffer_init(&cell->text);
    return cell;
}

void ucs_table_row_add_cell_empty(ucs_table_t *table, ucs_table_row_h row,
                                  unsigned col_span)
{
    ucs_table_row_add_cell(table, row, col_span, UCS_TABLE_ALIGN_LEFT);
}


void ucs_table_row_add_cell_fmt(ucs_table_t *table, ucs_table_row_h row,
                                unsigned col_span, ucs_table_align_t align,
                                const char *fmt, ...)
{
    ucs_table_cell_t *cell = ucs_table_row_add_cell(table, row, col_span,
                                                    align);
    const char UCS_V_UNUSED *cstr;
    va_list ap;

    va_start(ap, fmt);
    ucs_string_buffer_vappendf(&cell->text, fmt, ap);
    va_end(ap);

    cstr = ucs_string_buffer_cstr(&cell->text);

    ucs_assertv(strchr(cstr, '\n') == NULL,
                "table cell content must not contain '\\n': '%s'", cstr);
}


/* Calculate the total visible width of a cell spanning `col_span` columns. */
static unsigned ucs_table_cell_character_width(const unsigned *body_widths,
                                               unsigned start,
                                               unsigned col_span)
{
    unsigned width = 0;
    unsigned i;

    for (i = 0; i < col_span; ++i) {
        width += body_widths[start + i];
    }
    width += 3 * (col_span - 1);
    return width;
}


static unsigned ucs_table_cell_content_len(ucs_table_cell_t *cell)
{
    return ucs_string_buffer_length(&cell->text);
}


static void ucs_table_compute_widths(const ucs_table_t *table, unsigned *widths)
{
    ucs_table_entry_t *entry;
    ucs_table_cell_t *cell;
    unsigned i, body_col, content_len;
    unsigned existing;

    for (i = 0; i < table->config.n_cols; ++i) {
        widths[i] = 0;
    }

    /* Pass 1: col_span == 1 cells. */
    ucs_array_for_each(entry, &table->entries) {
        if (entry->kind != UCS_TABLE_ENTRY_ROW) {
            continue;
        }
        body_col = 0;
        ucs_array_for_each(cell, &entry->cells) {
            ucs_assertv(body_col + cell->col_span <= table->config.n_cols,
                        "row column span exceeds number of columns");

            if (cell->col_span == 1) {
                content_len      = ucs_table_cell_content_len(cell);
                widths[body_col] = ucs_max(widths[body_col], content_len);
            }
            body_col += cell->col_span;
        }
    }

    /* Pass 2: merged cells expand the rightmost spanned column. */
    ucs_array_for_each(entry, &table->entries) {
        if (entry->kind != UCS_TABLE_ENTRY_ROW) {
            continue;
        }

        body_col = 0;
        ucs_array_for_each(cell, &entry->cells) {
            ucs_assertv(body_col + cell->col_span <= table->config.n_cols,
                        "row column span exceeds number of columns");

            if (cell->col_span > 1) {
                content_len = ucs_table_cell_content_len(cell);
                existing    = ucs_table_cell_character_width(widths, body_col,
                                                             cell->col_span);

                if (content_len > existing) {
                    widths[body_col + cell->col_span - 1] += content_len -
                                                             existing;
                }
            }
            body_col += cell->col_span;
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


/* Render one row. The closing "|" has no trailing newline. */
static void ucs_table_render_cells(const ucs_table_t *table,
                                   const unsigned *widths,
                                   const ucs_table_cells_t *cells,
                                   ucs_string_buffer_t *strb)
{
    const ucs_table_cell_t *cell;
    unsigned body_col = 0;

    if (table->config.row_prefix != NULL) {
        ucs_string_buffer_appendf(strb, "%s", table->config.row_prefix);
    }

    ucs_array_for_each(cell, cells) {
        ucs_table_render_cell(strb, cell,
                              ucs_table_cell_character_width(widths, body_col,
                                                             cell->col_span));
        body_col += cell->col_span;
    }
    ucs_string_buffer_appendf(strb, "|");
}


static void ucs_table_render_separator(const ucs_table_t *table,
                                       const unsigned *widths,
                                       ucs_string_buffer_t *strb)
{
    unsigned i;

    if (table->config.row_prefix != NULL) {
        ucs_string_buffer_appendf(strb, "%s", table->config.row_prefix);
    }

    for (i = 0; i < table->config.n_cols; ++i) {
        ucs_string_buffer_appendc(strb, '+', 1);
        ucs_string_buffer_appendc(strb, '-', widths[i] + 2);
    }
    ucs_string_buffer_appendc(strb, '+', 1);
    ucs_string_buffer_appendc(strb, '\n', 1);
}


void ucs_table_render(ucs_table_t *table, ucs_string_buffer_t *strb)
{
    unsigned *widths = ucs_alloca(table->config.n_cols * sizeof(*widths));
    const ucs_table_entry_t *entry;
    unsigned i;

    ucs_table_compute_widths(table, widths);

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
}


void ucs_table_print(ucs_table_t *table)
{
    ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;

    ucs_table_render(table, &strb);
    printf("%s", ucs_string_buffer_cstr(&strb));
    ucs_string_buffer_cleanup(&strb);
}
