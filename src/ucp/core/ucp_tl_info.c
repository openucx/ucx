/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "ucp_tl_info.h"

#include <ucs/algorithm/qsort_r.h>
#include <ucs/datastruct/string_buffer.h>
#include <ucs/debug/log.h>
#include <ucs/debug/table.h>
#include <ucs/sys/math.h>
#include <ucs/sys/string.h>
#include <ucs/sys/topo/base/topo.h>
#include <string.h>


#define UCP_TL_INFO_DEVS_PER_LINE 3
#define UCP_TL_INFO_MARK_ENABLED  "+"
#define UCP_TL_INFO_MARK_DISABLED "-"
#define UCP_TL_INFO_HDR_TYPE      "Type"
#define UCP_TL_INFO_HDR_TRANSPORT "Transport"
#define UCP_TL_INFO_HDR_DEVICE    "Device (System device)"
#define UCP_TL_INFO_HDR_COMPONENT "Component"
#define UCP_TL_INFO_UNAVAILABLE   "<unavailable>"

#define UCP_TL_INFO_NUM_COLS 4


static const char *ucp_tl_info_legend_rows[] = {
    "Legend: + = enabled, - = disabled",
    "All of the available transports are listed, some may be disabled or unsupported on your system.",
    "All of the visible devices are listed per transport, some may be disabled.",
};


static int ucp_tl_info_is_same_group(const ucp_tl_info_entry_t *entries,
                                     unsigned a, unsigned b)
{
    return (entries[a].cmpt_index == entries[b].cmpt_index) &&
           (strcmp(entries[a].rsc.tl_name, entries[b].rsc.tl_name) == 0);
}

static int
ucp_tl_info_compare(const void *a, const void *b, void *UCS_V_UNUSED arg)
{
    const ucp_tl_info_entry_t *ea = a;
    const ucp_tl_info_entry_t *eb = b;
    int diff;

    diff = (int)ea->rsc.dev_type - (int)eb->rsc.dev_type;
    if (diff != 0) {
        return diff;
    }

    diff = (int)ea->cmpt_index - (int)eb->cmpt_index;
    if (diff != 0) {
        return diff;
    }

    return strcmp(ea->rsc.tl_name, eb->rsc.tl_name);
}

static int ucp_tl_info_cmpt_has_rscs(const ucp_tl_info_entry_t *all_rscs,
                                     unsigned num_all_rscs,
                                     ucp_rsc_index_t cmpt_idx)
{
    unsigned i;

    for (i = 0; i < num_all_rscs; ++i) {
        if (all_rscs[i].cmpt_index == cmpt_idx) {
            return 1;
        }
    }
    return 0;
}

/*
 * Emit one data row and clear the per-(type, cmpt, tl) "first" flags so
 * subsequent rows in the same group leave those columns blank.
 */
static void ucp_tl_info_emit_row(ucs_table_t *table, const char *type_str,
                                 const char *cmpt_str, const char *tl_str,
                                 const char *dev_str, int *first_type,
                                 int *first_cmpt, int *first_tl,
                                 int *printed_any)
{
    ucs_table_row_h row;

    ucs_table_add_row(table, &row);

    if (*first_type) {
        ucs_table_row_add_cell_fmt(table, row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                                   type_str);
    } else {
        ucs_table_row_add_cell_empty(table, row, 1);
    }
    if (*first_cmpt) {
        ucs_table_row_add_cell_fmt(table, row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                                   cmpt_str);
    } else {
        ucs_table_row_add_cell_empty(table, row, 1);
    }
    if (*first_tl) {
        ucs_table_row_add_cell_fmt(table, row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                                   tl_str);
    } else {
        ucs_table_row_add_cell_empty(table, row, 1);
    }
    ucs_table_row_add_cell_fmt(table, row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                               dev_str);

    *first_tl    = 0;
    *first_cmpt  = 0;
    *first_type  = 0;
    *printed_any = 1;
}

void ucp_context_log_tl_info(ucp_context_h context,
                             ucp_tl_info_entry_t *all_rscs,
                             unsigned num_all_rscs)
{
    ucs_string_buffer_t strb      = UCS_STRING_BUFFER_INITIALIZER;
    const ucs_table_config_t tcfg = {
        .n_cols = UCP_TL_INFO_NUM_COLS
    };
    UCS_STRING_BUFFER_ONSTACK(tl_strb, UCT_TL_NAME_MAX + 8);
    UCS_STRING_BUFFER_ONSTACK(dev_strb, 512);
    UCS_STRING_BUFFER_ONSTACK(title_strb, 128);
    int printed_any, first_type, first_cmpt, first_tl, first_unavail;
    int tl_enabled, dev_count, new_dev_type, new_cmpt;
    uct_device_type_t dev_type;
    ucp_rsc_index_t cmpt_idx;
    ucs_table_row_h row;
    ucs_table_t table;
    ucs_status_t status;
    unsigned i, j, group_end;

    if (!context->config.ext.print_transport_tables) {
        return;
    }

    if ((all_rscs == NULL) || (num_all_rscs == 0)) {
        ucs_warn("skipping transport table: no resource info captured");
        return;
    }

    /* Sort by (dev_type, component, transport) so that the rows of each group
     * are contiguous and group boundaries can be found by comparing adjacent
     * entries, instead of rescanning the whole array per group. */
    ucs_qsort_r(all_rscs, num_all_rscs, sizeof(*all_rscs), ucp_tl_info_compare,
                NULL);

    ucs_table_init(&table, &tcfg);

    ucs_string_buffer_appendf(&title_strb, "Available Transports and Devices");
    if (!ucs_string_is_empty(context->name)) {
        ucs_string_buffer_appendf(&title_strb, " (ctx: %s)", context->name);
    }

    /* Title spans all body columns. */
    ucs_table_add_row(&table, &row);
    ucs_table_row_add_cell_fmt(&table, row, UCP_TL_INFO_NUM_COLS,
                               UCS_TABLE_ALIGN_LEFT, "%s",
                               ucs_string_buffer_cstr(&title_strb));
    ucs_table_add_separator(&table);

    /* Column headers. */
    ucs_table_add_row(&table, &row);
    ucs_table_row_add_cell_fmt(&table, row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                               UCP_TL_INFO_HDR_TYPE);
    ucs_table_row_add_cell_fmt(&table, row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                               UCP_TL_INFO_HDR_COMPONENT);
    ucs_table_row_add_cell_fmt(&table, row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                               UCP_TL_INFO_HDR_TRANSPORT);
    ucs_table_row_add_cell_fmt(&table, row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                               UCP_TL_INFO_HDR_DEVICE);
    ucs_table_add_separator(&table);

    printed_any = 0;
    for (i = 0; i < num_all_rscs; i = group_end) {
        /* The sorted array keeps each (component, transport) group contiguous;
         * find the end of the group that starts at i. */
        for (group_end = i + 1; group_end < num_all_rscs; ++group_end) {
            if (!ucp_tl_info_is_same_group(all_rscs, i, group_end)) {
                break;
            }
        }

        dev_type = all_rscs[i].rsc.dev_type;
        cmpt_idx = all_rscs[i].cmpt_index;

        new_dev_type = (i == 0) || (all_rscs[i - 1].rsc.dev_type != dev_type);
        new_cmpt   = new_dev_type || (all_rscs[i - 1].cmpt_index != cmpt_idx);
        first_type = new_dev_type;
        first_cmpt = new_cmpt;
        first_tl   = 1;

        if (printed_any) {
            /* Carry-over cols on the separator above this row:
             * 2 for a new TL in the same component, 1 for a
             * new component in the same dev_type, 0 otherwise. */
            ucs_table_add_separator_with_merged_cols(
                    &table, !new_cmpt ? 2 : (new_dev_type ? 0 : 1));
        }

        tl_enabled = 0;
        for (j = i; j < group_end; ++j) {
            if (all_rscs[j].enabled) {
                tl_enabled = 1;
                break;
            }
        }

        ucs_string_buffer_reset(&tl_strb);
        ucs_string_buffer_appendf(&tl_strb, "%s %s",
                                  tl_enabled ? UCP_TL_INFO_MARK_ENABLED :
                                               UCP_TL_INFO_MARK_DISABLED,
                                  all_rscs[i].rsc.tl_name);

        dev_count = 0;
        ucs_string_buffer_reset(&dev_strb);
        for (j = i; j < group_end; ++j) {
            if ((dev_count > 0) &&
                (dev_count % UCP_TL_INFO_DEVS_PER_LINE == 0)) {
                ucp_tl_info_emit_row(&table, uct_device_type_names[dev_type],
                                     context->tl_cmpts[cmpt_idx].attr.name,
                                     ucs_string_buffer_cstr(&tl_strb),
                                     ucs_string_buffer_cstr(&dev_strb),
                                     &first_type, &first_cmpt, &first_tl,
                                     &printed_any);
                ucs_string_buffer_reset(&dev_strb);
            }

            if (dev_count % UCP_TL_INFO_DEVS_PER_LINE > 0) {
                ucs_string_buffer_appendf(&dev_strb, "  ");
            }

            if (all_rscs[j].rsc.sys_device != UCS_SYS_DEVICE_ID_UNKNOWN) {
                ucs_string_buffer_appendf(&dev_strb, "%s %s (%s)",
                                          all_rscs[j].enabled ?
                                                  UCP_TL_INFO_MARK_ENABLED :
                                                  UCP_TL_INFO_MARK_DISABLED,
                                          all_rscs[j].rsc.dev_name,
                                          ucs_topo_sys_device_get_name(
                                                  all_rscs[j].rsc.sys_device));
            } else {
                ucs_string_buffer_appendf(&dev_strb, "%s %s",
                                          all_rscs[j].enabled ?
                                                  UCP_TL_INFO_MARK_ENABLED :
                                                  UCP_TL_INFO_MARK_DISABLED,
                                          all_rscs[j].rsc.dev_name);
            }

            dev_count++;
        }

        if (ucs_string_buffer_length(&dev_strb) > 0) {
            ucp_tl_info_emit_row(&table, uct_device_type_names[dev_type],
                                 context->tl_cmpts[cmpt_idx].attr.name,
                                 ucs_string_buffer_cstr(&tl_strb),
                                 ucs_string_buffer_cstr(&dev_strb), &first_type,
                                 &first_cmpt, &first_tl, &printed_any);
        }
    }

    first_unavail = 1;
    for (cmpt_idx = 0; cmpt_idx < context->num_cmpts; ++cmpt_idx) {
        if (!ucp_tl_info_cmpt_has_rscs(all_rscs, num_all_rscs, cmpt_idx)) {
            if (first_unavail) {
                if (printed_any) {
                    ucs_table_add_separator(&table);
                }
                ucs_table_add_row(&table, &row);
                ucs_table_row_add_cell_fmt(&table, row, 1, UCS_TABLE_ALIGN_LEFT,
                                           "%s", UCP_TL_INFO_UNAVAILABLE);
                ucs_table_row_add_cell_fmt(
                        &table, row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                        context->tl_cmpts[cmpt_idx].attr.name);
                ucs_table_row_add_cell_empty(&table, row, 1);
                ucs_table_row_add_cell_empty(&table, row, 1);
                first_unavail = 0;
            } else {
                /* Carry over the empty "type" column with merged_cols=1
                 * so the separator stays blank above it. */
                ucs_table_add_separator_with_merged_cols(&table, 1);
                ucs_table_add_row(&table, &row);
                ucs_table_row_add_cell_empty(&table, row, 1);
                ucs_table_row_add_cell_fmt(
                        &table, row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                        context->tl_cmpts[cmpt_idx].attr.name);
                ucs_table_row_add_cell_empty(&table, row, 1);
                ucs_table_row_add_cell_empty(&table, row, 1);
            }
            printed_any = 1;
        }
    }

    ucs_table_add_separator(&table);
    for (i = 0; i < ucs_static_array_size(ucp_tl_info_legend_rows); ++i) {
        ucs_table_add_row(&table, &row);
        ucs_table_row_add_cell_fmt(&table, row, UCP_TL_INFO_NUM_COLS,
                                   UCS_TABLE_ALIGN_LEFT, "%s",
                                   ucp_tl_info_legend_rows[i]);
    }

    ucs_table_render(&table, &strb);

    status = ucs_table_get_status(&table);
    if (status != UCS_OK) {
        ucs_warn("transport table render incomplete: %s",
                 ucs_status_string(status));
    }

    ucs_log_print_compact(ucs_string_buffer_cstr(&strb));
    ucs_string_buffer_cleanup(&strb);
    ucs_table_cleanup(&table);
}
