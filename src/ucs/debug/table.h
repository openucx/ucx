/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_TABLE_H_
#define UCS_TABLE_H_

#include <ucs/datastruct/array.h>
#include <ucs/datastruct/string_buffer.h>
#include <ucs/sys/compiler_def.h>


BEGIN_C_DECLS

typedef unsigned ucs_table_row_h;


/**
 * Cell alignment, fixed at add-cell time.
 */
typedef enum {
    UCS_TABLE_ALIGN_LEFT, /**< pad on the right */
    UCS_TABLE_ALIGN_RIGHT, /**< pad on the left */
    UCS_TABLE_ALIGN_CENTER /**< pad equally (right side gets the extra
                                 when padding is odd) */
} ucs_table_align_t;


/** Cell entry (internal). */
typedef struct ucs_table_cell {
    unsigned            col_span; /**< body columns spanned */
    ucs_table_align_t   align; /**< alignment selected at add-cell time */
    ucs_string_buffer_t text; /**< cell content, owned by the table */
} ucs_table_cell_t;


typedef enum {
    UCS_TABLE_ENTRY_ROW,
    UCS_TABLE_ENTRY_SEPARATOR
} ucs_table_entry_kind_t;


/** Named dynamic array of cells used by a row entry. */
UCS_ARRAY_DECLARE_TYPE(ucs_table_cells_t, unsigned, ucs_table_cell_t);


/**
 * Table entry: a row (vector of cells) or a separator. `cells` is only
 * populated when `kind == UCS_TABLE_ENTRY_ROW`.
 */
typedef struct {
    ucs_table_entry_kind_t kind;
    ucs_table_cells_t      cells;
} ucs_table_entry_t;


UCS_ARRAY_DECLARE_TYPE(ucs_table_entries_t, unsigned, ucs_table_entry_t);


/**
 * Configuration for `ucs_table_init()`. Zero-initialize and set the fields
 * you need; only `n_cols` is required.
 */
typedef struct ucs_table_config {
    /** Total number of columns; per-row cells' col_spans must sum to this. */
    unsigned   n_cols;
    /** Prepended to every rendered line */
    const char *row_prefix;
    /** When non-zero, render every column at the maximum computed width so
     *  all columns are equal-width. */
    int        equal_widths;
} ucs_table_config_t;


/**
 * Buffered ASCII table.
 */
typedef struct ucs_table {
    ucs_table_config_t  config;
    ucs_table_entries_t entries;
} ucs_table_t;


/**
 * Initialize a buffered table.
 *
 * @param [out] table   Table to initialize.
 * @param [in]  config  Configuration (non-NULL); see ucs_table_config_t.
 *                      Copied into the table.
 */
void ucs_table_init(ucs_table_t *table, const ucs_table_config_t *config);


/**
 * Release all storage owned by the table. After this call the table is
 * unusable.
 *
 * @param [in,out] table  Table to clean up.
 */
void ucs_table_cleanup(ucs_table_t *table);


/**
 * Append a horizontal separator. The top and bottom frame separators are
 * inserted automatically by ucs_table_render();
 *
 * @param [in,out] table  Table to append to.
 */
void ucs_table_add_separator(ucs_table_t *table);


/**
 * Begin a new row. Subsequent ucs_table_row_add_cell_*() calls populate it
 * left-to-right; the sum of col_spans must equal n_cols. The returned
 * handle is valid until the table is cleaned up;
 *
 * @param [in,out] table  Table to append to.
 * @return Row handle for use with add-cell functions.
 */
ucs_table_row_h ucs_table_add_row(ucs_table_t *table);


/**
 * Add an empty cell.
 *
 * @param [in,out] table     Table that owns @a row.
 * @param [in]     row       Row returned by ucs_table_add_row().
 * @param [in]     col_span  Number of body columns to span.
 */
void ucs_table_row_add_cell_empty(ucs_table_t *table, ucs_table_row_h row,
                                  unsigned col_span);


/**
 * Add a cell with printf-style content. Asserts the result has no '\n'.
 *
 * @param [in,out] table     Table that owns @a row.
 * @param [in]     row       Row returned by ucs_table_add_row().
 * @param [in]     col_span  Number of body columns to span.
 * @param [in]     align     Cell alignment.
 * @param [in]     fmt       printf format string.
 */
void ucs_table_row_add_cell_fmt(ucs_table_t *table, ucs_table_row_h row,
                                unsigned col_span, ucs_table_align_t align,
                                const char *fmt, ...) UCS_F_PRINTF(5, 6);


/**
 * Render the table into @a strb. Recomputes column widths on every call,
 * adapting to any rows or separators added since the previous render, and
 * emits top frame + body rows/separators + bottom frame.
 *
 * @param [in,out] table  Table to render.
 * @param [in,out] strb   Destination string buffer.
 */
void ucs_table_render(ucs_table_t *table, ucs_string_buffer_t *strb);


/**
 * Render the table directly to stdout.
 *
 * @param [in,out] table  Table to print.
 */
void ucs_table_print(ucs_table_t *table);


END_C_DECLS

#endif
