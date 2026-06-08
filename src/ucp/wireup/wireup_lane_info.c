/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "wireup_lane_info.h"

#include <ucp/proto/lane_type.h>
#include <ucs/datastruct/string_buffer.h>
#include <ucs/debug/log.h>
#include <ucs/debug/table.h>
#include <ucs/sys/string.h>
#include <ucs/sys/topo/base/topo.h>
#include <string.h>


#define UCP_EP_LANE_INFO_HDR_TL    "Transport"
#define UCP_EP_LANE_INFO_HDR_DEV   "Device (Sys. dev.)"
#define UCP_EP_LANE_INFO_HDR_COUNT "# Lanes"
#define UCP_EP_LANE_INFO_HDR_TYPES "Lane Types"

#define UCP_EP_LANE_INFO_NUM_COLS 4


static int ucp_ep_lane_is_same_dev(const ucp_ep_config_key_t *key,
                                   ucp_lane_index_t a, ucp_lane_index_t b)
{
    if ((a == key->cm_lane) || (b == key->cm_lane)) {
        return a == b;
    }
    return key->lanes[a].rsc_index == key->lanes[b].rsc_index;
}

static int
ucp_ep_lane_is_dev_leader(const ucp_ep_config_key_t *key, ucp_lane_index_t lane)
{
    ucp_lane_index_t j;

    for (j = 0; j < lane; ++j) {
        if (ucp_ep_lane_is_same_dev(key, j, lane)) {
            return 0;
        }
    }
    return 1;
}

static int ucp_ep_lane_is_same_tl(const ucp_ep_config_key_t *key,
                                  ucp_context_h context, ucp_lane_index_t a,
                                  ucp_lane_index_t b)
{
    if ((a == key->cm_lane) || (b == key->cm_lane)) {
        return a == b;
    }
    return strcmp(context->tl_rscs[key->lanes[a].rsc_index].tl_rsc.tl_name,
                  context->tl_rscs[key->lanes[b].rsc_index].tl_rsc.tl_name) ==
           0;
}

static void
ucp_wireup_get_lane_names(const ucp_ep_config_key_t *key, ucp_context_h context,
                          ucp_lane_index_t lane, const char **tl_name_p,
                          const char **dev_name_p)
{
    if (lane == key->cm_lane) {
        *tl_name_p  = "cm";
        *dev_name_p = "cm";
    } else {
        *tl_name_p = context->tl_rscs[key->lanes[lane].rsc_index].tl_rsc.tl_name;
        *dev_name_p =
                context->tl_rscs[key->lanes[lane].rsc_index].tl_rsc.dev_name;
    }
}

static void
ucp_wireup_format_lane_dev(const ucp_ep_config_key_t *key,
                           ucp_context_h context, ucp_lane_index_t lane,
                           const char *dev_name, ucs_string_buffer_t *strb)
{
    const uct_tl_resource_desc_t *rsc;
    const char *sysdev_name = NULL;

    if (lane != key->cm_lane) {
        rsc = &context->tl_rscs[key->lanes[lane].rsc_index].tl_rsc;
        if (rsc->sys_device != UCS_SYS_DEVICE_ID_UNKNOWN) {
            sysdev_name = ucs_topo_sys_device_get_name(rsc->sys_device);
        }
    }

    if (sysdev_name != NULL) {
        ucs_string_buffer_appendf(strb, "%s (%s)", dev_name, sysdev_name);
    } else {
        ucs_string_buffer_appendf(strb, "%s", dev_name);
    }
}

static void ucp_wireup_format_lane_types(ucp_lane_type_mask_t types_union,
                                         ucs_string_buffer_t *strb)
{
    int first = 1;
    ucp_lane_type_t lt;

    for (lt = UCP_LANE_TYPE_FIRST; lt < UCP_LANE_TYPE_LAST; ++lt) {
        if (!(types_union & UCS_BIT(lt)) ||
            (ucp_lane_type_info[lt].short_name == NULL)) {
            continue;
        }

        if (!first) {
            ucs_string_buffer_appendf(strb, ", ");
        }

        ucs_string_buffer_appendf(strb, "%s",
                                  ucp_lane_type_info[lt].short_name);
        first = 0;
    }
}

static ucp_lane_type_mask_t
ucp_wireup_collect_lane_types(const ucp_ep_config_key_t *key,
                              ucp_lane_index_t leader, int *count_p)
{
    ucp_lane_type_mask_t types_union = 0;
    ucp_lane_index_t j;
    int count = 0;

    for (j = 0; j < key->num_lanes; ++j) {
        if (ucp_ep_lane_is_same_dev(key, leader, j)) {
            count++;
            types_union |= key->lanes[j].lane_types;
        }
    }

    if (count_p != NULL) {
        *count_p = count;
    }

    return types_union;
}

void ucp_wireup_log_ep_lanes(ucp_worker_h worker,
                             const ucp_ep_config_key_t *key,
                             ucp_worker_cfg_index_t cfg_index)
{
    ucp_context_h context         = worker->context;
    ucs_string_buffer_t strb      = UCS_STRING_BUFFER_INITIALIZER;
    const ucs_table_config_t tcfg = {
        .n_cols = UCP_EP_LANE_INFO_NUM_COLS
    };
    UCS_STRING_BUFFER_ONSTACK(title_strb, 128);
    UCS_STRING_BUFFER_ONSTACK(types_strb, 128);
    UCS_STRING_BUFFER_ONSTACK(dev_strb, 128);
    ucs_table_t table;
    ucs_table_row_h row;
    ucp_lane_index_t lane, j;
    ucp_lane_type_mask_t types_union;
    const char *tl_name, *dev_name, *ep_type;
    ucs_status_t status;
    int count, first_tl, printed_any;

    if (!context->config.ext.print_transport_tables) {
        return;
    }

    if (key->flags & UCP_EP_CONFIG_KEY_FLAG_SELF) {
        ep_type = "self";
    } else if (key->flags & UCP_EP_CONFIG_KEY_FLAG_INTRA_NODE) {
        ep_type = "intra-node";
    } else {
        ep_type = "inter-node";
    }

    ucs_string_buffer_appendf(&title_strb, "Endpoint Config #%d (", cfg_index);
    if (!ucs_string_is_empty(context->name)) {
        ucs_string_buffer_appendf(&title_strb, "ctx: %s, ", context->name);
    }
    ucs_string_buffer_appendf(&title_strb, "type: %s)", ep_type);

    ucs_table_init(&table, &tcfg);

    /* Title spans all body columns. */
    ucs_table_add_row(&table, &row);
    ucs_table_row_add_cell_fmt(&table, row, UCP_EP_LANE_INFO_NUM_COLS,
                               UCS_TABLE_ALIGN_LEFT, "%s",
                               ucs_string_buffer_cstr(&title_strb));
    ucs_table_add_separator(&table);

    /* Column headers. */
    ucs_table_add_row(&table, &row);
    ucs_table_row_add_cell_fmt(&table, row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                               UCP_EP_LANE_INFO_HDR_TL);
    ucs_table_row_add_cell_fmt(&table, row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                               UCP_EP_LANE_INFO_HDR_DEV);
    ucs_table_row_add_cell_fmt(&table, row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                               UCP_EP_LANE_INFO_HDR_COUNT);
    ucs_table_row_add_cell_fmt(&table, row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                               UCP_EP_LANE_INFO_HDR_TYPES);
    ucs_table_add_separator(&table);

    printed_any = 0;
    for (lane = 0; lane < key->num_lanes; ++lane) {
        if (!ucp_ep_lane_is_dev_leader(key, lane)) {
            continue;
        }

        first_tl = 1;
        for (j = 0; j < lane; ++j) {
            if (!ucp_ep_lane_is_dev_leader(key, j)) {
                continue;
            }
            if (ucp_ep_lane_is_same_tl(key, context, j, lane)) {
                first_tl = 0;
                break;
            }
        }

        if (first_tl && printed_any) {
            ucs_table_add_separator(&table);
        }

        ucp_wireup_get_lane_names(key, context, lane, &tl_name, &dev_name);

        ucs_string_buffer_reset(&dev_strb);
        ucp_wireup_format_lane_dev(key, context, lane, dev_name, &dev_strb);

        types_union = ucp_wireup_collect_lane_types(key, lane, &count);
        ucs_string_buffer_reset(&types_strb);
        ucp_wireup_format_lane_types(types_union, &types_strb);

        ucs_table_add_row(&table, &row);
        ucs_table_row_add_cell_fmt(&table, row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                                   first_tl ? tl_name : "");
        ucs_table_row_add_cell_fmt(&table, row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                                   ucs_string_buffer_cstr(&dev_strb));
        ucs_table_row_add_cell_fmt(&table, row, 1, UCS_TABLE_ALIGN_LEFT, "%d",
                                   count);
        ucs_table_row_add_cell_fmt(&table, row, 1, UCS_TABLE_ALIGN_LEFT, "%s",
                                   ucs_string_buffer_cstr(&types_strb));

        printed_any = 1;
    }

    ucs_table_render(&table, &strb);

    status = ucs_table_get_status(&table);
    if (status != UCS_OK) {
        ucs_warn("endpoint lane table render incomplete: %s",
                 ucs_status_string(status));
    }

    ucs_log_print_compact(ucs_string_buffer_cstr(&strb));
    ucs_string_buffer_cleanup(&strb);
    ucs_table_cleanup(&table);
}
