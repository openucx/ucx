/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "ucp_tl_info.h"

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
ucp_tl_info_is_group_leader(const ucp_tl_info_entry_t *entries, unsigned idx)
{
    unsigned j;

    for (j = 0; j < idx; ++j) {
        if (ucp_tl_info_is_same_group(entries, j, idx)) {
            return 0;
        }
    }
    return 1;
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

static uct_device_type_t
ucp_tl_info_cmpt_dev_type(const ucp_tl_info_entry_t *all_rscs,
                          unsigned num_all_rscs, ucp_rsc_index_t cmpt_idx)
{
    unsigned i;

    for (i = 0; i < num_all_rscs; ++i) {
        if (all_rscs[i].cmpt_index == cmpt_idx) {
            return all_rscs[i].rsc.dev_type;
        }
    }
    return UCT_DEVICE_TYPE_LAST;
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
                             const ucp_tl_info_entry_t *all_rscs,
                             unsigned num_all_rscs)
{
    ucs_string_buffer_t strb      = UCS_STRING_BUFFER_INITIALIZER;
    const ucs_table_config_t tcfg = {
        .n_cols = UCP_TL_INFO_NUM_COLS
    };
    ucs_table_t table;
    ucs_table_row_h row;
    ucp_rsc_index_t cmpt_idx;
    uct_device_type_t dev_type, cmpt_dev_type;
    unsigned i, j;
    size_t dev_buf_len;
    int printed_any, first_type, first_cmpt, first_tl, first_unavail;
    int dev_count, tl_enabled;
    char dev_buf[512];
    char title_buf[96];
    char tl_buf[UCT_TL_NAME_MAX + 8];

    if (!context->config.ext.print_transport_tables) {
        return;
    }

    ucs_assertv(all_rscs != NULL, "all_rscs must not be NULL");
    ucs_assertv(num_all_rscs > 0, "num_all_rscs must be greater than 0");

    ucs_table_init(&table, &tcfg);

    if (!ucs_string_is_empty(context->name)) {
        snprintf(title_buf, sizeof(title_buf),
                 "Available Transports and Devices (ctx: %s)", context->name);
    } else {
        snprintf(title_buf, sizeof(title_buf),
                 "Available Transports and Devices");
    }

    /* Title spans all body columns. */
    ucs_table_add_row(&table, &row);
    ucs_table_row_add_cell_fmt(&table, row, UCP_TL_INFO_NUM_COLS,
                               UCS_TABLE_ALIGN_LEFT, "%s", title_buf);
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
    for (dev_type = UCT_DEVICE_TYPE_NET; dev_type < UCT_DEVICE_TYPE_LAST;
         ++dev_type) {
        first_type = 1;
        for (cmpt_idx = 0; cmpt_idx < context->num_cmpts; ++cmpt_idx) {
            /* All resources from a single component are assumed to share the
             * same device type, so the first match determines the type */
            cmpt_dev_type = ucp_tl_info_cmpt_dev_type(all_rscs, num_all_rscs,
                                                      cmpt_idx);
            if (cmpt_dev_type != dev_type) {
                continue;
            }

            first_cmpt = 1;
            for (i = 0; i < num_all_rscs; ++i) {
                if ((all_rscs[i].cmpt_index != cmpt_idx) ||
                    !ucp_tl_info_is_group_leader(all_rscs, i)) {
                    continue;
                }

                if (printed_any) {
                    /* Carry-over cols on the separator above this row:
                     * 2 for a new TL in the same component, 1 for a
                     * new component in the same dev_type, 0 otherwise. */
                    ucs_table_add_separator_with_merged_cols(
                            &table, !first_cmpt ? 2 : (first_type ? 0 : 1));
                }

                tl_enabled = 0;
                for (j = i; j < num_all_rscs; ++j) {
                    if (ucp_tl_info_is_same_group(all_rscs, j, i) &&
                        all_rscs[j].enabled) {
                        tl_enabled = 1;
                        break;
                    }
                }

                snprintf(tl_buf, sizeof(tl_buf), "%s %s",
                         tl_enabled ? UCP_TL_INFO_MARK_ENABLED :
                                      UCP_TL_INFO_MARK_DISABLED,
                         all_rscs[i].rsc.tl_name);

                first_tl    = 1;
                dev_count   = 0;
                dev_buf[0]  = '\0';
                dev_buf_len = 0;
                for (j = i; j < num_all_rscs; ++j) {
                    if (!ucp_tl_info_is_same_group(all_rscs, j, i)) {
                        continue;
                    }

                    if ((dev_count > 0) &&
                        (dev_count % UCP_TL_INFO_DEVS_PER_LINE == 0)) {
                        ucp_tl_info_emit_row(
                                &table, uct_device_type_names[dev_type],
                                context->tl_cmpts[cmpt_idx].attr.name, tl_buf,
                                dev_buf, &first_type, &first_cmpt, &first_tl,
                                &printed_any);
                        dev_buf[0]  = '\0';
                        dev_buf_len = 0;
                    }

                    if (dev_count % UCP_TL_INFO_DEVS_PER_LINE > 0) {
                        dev_buf_len += snprintf(dev_buf + dev_buf_len,
                                                sizeof(dev_buf) - dev_buf_len,
                                                "  ");
                        if (dev_buf_len >= sizeof(dev_buf)) {
                            dev_buf_len = sizeof(dev_buf) - 1;
                        }
                    }
                    if (all_rscs[j].rsc.sys_device !=
                        UCS_SYS_DEVICE_ID_UNKNOWN) {
                        dev_buf_len += snprintf(
                                dev_buf + dev_buf_len,
                                sizeof(dev_buf) - dev_buf_len, "%s %s (%s)",
                                all_rscs[j].enabled ? UCP_TL_INFO_MARK_ENABLED :
                                                      UCP_TL_INFO_MARK_DISABLED,
                                all_rscs[j].rsc.dev_name,
                                ucs_topo_sys_device_get_name(
                                        all_rscs[j].rsc.sys_device));
                    } else {
                        dev_buf_len += snprintf(
                                dev_buf + dev_buf_len,
                                sizeof(dev_buf) - dev_buf_len, "%s %s",
                                all_rscs[j].enabled ? UCP_TL_INFO_MARK_ENABLED :
                                                      UCP_TL_INFO_MARK_DISABLED,
                                all_rscs[j].rsc.dev_name);
                    }
                    if (dev_buf_len >= sizeof(dev_buf)) {
                        dev_buf_len = sizeof(dev_buf) - 1;
                    }
                    dev_count++;
                }

                if (dev_buf[0] != '\0') {
                    ucp_tl_info_emit_row(&table,
                                         uct_device_type_names[dev_type],
                                         context->tl_cmpts[cmpt_idx].attr.name,
                                         tl_buf, dev_buf, &first_type,
                                         &first_cmpt, &first_tl, &printed_any);
                }
            }
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

    /* Legend information */
    ucs_table_add_separator(&table);
    for (i = 0; i < ucs_static_array_size(ucp_tl_info_legend_rows); ++i) {
        ucs_table_add_row(&table, &row);
        ucs_table_row_add_cell_fmt(&table, row, UCP_TL_INFO_NUM_COLS,
                                   UCS_TABLE_ALIGN_LEFT, "%s",
                                   ucp_tl_info_legend_rows[i]);
    }

    ucs_table_render(&table, &strb);
    ucs_log_print_compact(ucs_string_buffer_cstr(&strb));
    ucs_string_buffer_cleanup(&strb);
    ucs_table_cleanup(&table);
}
