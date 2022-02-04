/*
 * Copyright (C) 2022, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "proto_debug.h"
#include "proto_init.h"
#include "proto_select.inl"

#include <ucs/arch/atomic.h>
#include <ucs/datastruct/array.inl>


/* Protocol information table */
typedef struct {
    char range_str[32];
    char desc[UCP_PROTO_DESC_STR_MAX];
    char config[UCP_PROTO_CONFIG_STR_MAX];
} ucp_proto_info_row_t;
UCS_ARRAY_DEFINE_INLINE(ucp_proto_info_table, unsigned, ucp_proto_info_row_t);


void ucp_proto_select_perf_str(const ucs_linear_func_t *perf, char *time_str,
                               size_t time_str_max, char *bw_str,
                               size_t bw_str_max)
{
    /* Estimated time per 1024 bytes */
    ucs_snprintf_safe(time_str, time_str_max, UCP_PROTO_PERF_FUNC_TIME_FMT,
                      UCP_PROTO_PERF_FUNC_TIME_ARG(perf));

    /* Estimated bandwidth (MiB/s) */
    ucs_snprintf_safe(bw_str, bw_str_max, UCP_PROTO_PERF_FUNC_BW_FMT,
                      UCP_PROTO_PERF_FUNC_BW_ARG(perf));
}

void ucp_proto_select_init_trace_caps(
        ucp_proto_id_t proto_id, const ucp_proto_init_params_t *init_params)
{
    ucp_proto_caps_t *proto_caps          = init_params->caps;
    ucp_proto_query_params_t query_params = {
        .proto         = ucp_protocols[proto_id],
        .priv          = init_params->priv,
        .worker        = init_params->worker,
        .select_param  = init_params->select_param,
        .ep_config_key = init_params->ep_config_key,
        .msg_length    = proto_caps->min_length
    };
    const UCS_V_UNUSED ucs_linear_func_t *perf;
    size_t UCS_V_UNUSED range_start, range_end;
    ucp_proto_query_attr_t query_attr;
    int UCS_V_UNUSED range_index;
    char min_length_str[64];
    char thresh_str[64];

    if (!ucs_log_is_enabled(UCS_LOG_LEVEL_TRACE)) {
        return;
    }

    ucs_trace("initialized protocol %s min_length %s cfg_thresh %s",
              init_params->proto_name,
              ucs_memunits_to_str(proto_caps->min_length, min_length_str,
                                  sizeof(min_length_str)),
              ucs_memunits_to_str(proto_caps->cfg_thresh, thresh_str,
                                  sizeof(thresh_str)));

    ucs_log_indent(1);
    range_start = 0;
    for (range_index = 0; range_index < proto_caps->num_ranges; ++range_index) {
        range_start = ucs_max(range_start, proto_caps->min_length);
        range_end   = proto_caps->ranges[range_index].max_length;
        if (range_end > range_start) {
            query_params.msg_length = range_start;

            ucp_proto_id_call(proto_id, query, &query_params, &query_attr);

            perf = proto_caps->ranges[range_index].perf;
            ucs_trace("range[%d] %s %s %s" UCP_PROTO_PERF_FUNC_TYPES_FMT,
                      range_index, query_attr.desc, query_attr.config,
                      ucs_memunits_range_str(range_start, range_end, thresh_str,
                                             sizeof(thresh_str)),
                      UCP_PROTO_PERF_FUNC_TYPES_ARG(perf));
        }
        range_start = range_end + 1;
    }
    ucs_log_indent(-1);
}

static void
ucp_proto_select_param_dump(ucp_worker_h worker,
                            ucp_worker_cfg_index_t ep_cfg_index,
                            ucp_worker_cfg_index_t rkey_cfg_index,
                            const ucp_proto_select_param_t *select_param,
                            const char **operation_names,
                            ucs_string_buffer_t *ep_cfg_strb,
                            ucs_string_buffer_t *select_param_strb)
{
    if (!ucs_string_is_empty(worker->context->name)) {
        ucs_string_buffer_appendf(ep_cfg_strb, "%s ", worker->context->name);
    }
    ucs_string_buffer_appendf(ep_cfg_strb, "ep_cfg[%d]", ep_cfg_index);

    /* Operation name and attributes */
    ucp_proto_select_info_str(worker, rkey_cfg_index, select_param,
                              operation_names, select_param_strb);
}

static double
ucp_proto_select_calc_bandwidth(const ucp_proto_select_param_t *select_param,
                                const ucp_proto_perf_range_t *range,
                                size_t msg_length)
{
    const ucs_linear_func_t *perf_func;
    ucp_proto_perf_type_t perf_type;

    perf_type = ucp_proto_select_param_perf_type(select_param);
    perf_func = &range->perf[perf_type];

    return msg_length / ucs_linear_func_apply(*perf_func, msg_length);
}

static void ucp_proto_table_row_separator(ucs_string_buffer_t *strb,
                                          const int *column_width,
                                          unsigned num_columns)
{
    unsigned i;

    ucs_string_buffer_appendc(strb, '+', 1);
    for (i = 0; i < num_columns; ++i) {
        ucs_string_buffer_appendc(strb, '-', column_width[i] + 2);
        ucs_string_buffer_appendc(strb, '+', 1);
    }
    ucs_string_buffer_appendc(strb, '\n', 1);
}

static void
ucp_proto_select_elem_info(ucp_worker_h worker,
                           ucp_worker_cfg_index_t ep_cfg_index,
                           ucp_worker_cfg_index_t rkey_cfg_index,
                           const ucp_proto_select_param_t *select_param,
                           ucp_proto_select_elem_t *select_elem,
                           ucs_string_buffer_t *strb)
{
    UCS_STRING_BUFFER_ONSTACK(ep_cfg_strb, UCP_PROTO_CONFIG_STR_MAX);
    UCS_STRING_BUFFER_ONSTACK(select_param_strb, UCP_PROTO_CONFIG_STR_MAX);
    static const char *info_row_fmt = "| %*s | %-*s | %-*s |\n";
    ucs_array_t(ucp_proto_info_table) table;
    int hdr_col_width[2], col_width[3];
    ucp_proto_query_attr_t proto_attr;
    ucp_proto_info_row_t *row_elem;
    size_t range_start, range_end;
    ucs_status_t status;
    int proto_valid;

    ucp_proto_select_param_dump(worker, ep_cfg_index, rkey_cfg_index,
                                select_param, ucp_operation_descs, &ep_cfg_strb,
                                &select_param_strb);

    /* Populate the table and column widths */
    ucs_array_init_dynamic(&table);
    col_width[0] = ucs_string_buffer_length(&ep_cfg_strb);
    col_width[1] = 0;
    col_width[2] = 0;
    range_end    = -1;
    do {
        range_start = range_end + 1;

        proto_valid = ucp_proto_select_elem_query(worker, select_elem,
                                                  range_start, &proto_attr);
        range_end   = proto_attr.max_msg_length;
        if (!proto_valid) {
            continue;
        }

        status = ucs_array_append(ucp_proto_info_table, &table);
        if (status != UCS_OK) {
            break;
        }

        row_elem = ucs_array_last(&table);
        ucs_snprintf_safe(row_elem->desc, sizeof(row_elem->desc), "%s%s",
                          proto_attr.is_estimation ? "(?) " : "",
                          proto_attr.desc);
        ucs_strncpy_safe(row_elem->config, proto_attr.config,
                         sizeof(row_elem->config));

        ucs_memunits_range_str(range_start, range_end, row_elem->range_str,
                               sizeof(row_elem->range_str));

        col_width[0] = ucs_max(col_width[0], strlen(row_elem->range_str));
        col_width[1] = ucs_max(col_width[1], strlen(row_elem->desc));
        col_width[2] = ucs_max(col_width[2], strlen(row_elem->config));
    } while (range_end != SIZE_MAX);

    /* Resize column[1] to match longest row including header */
    col_width[1] = ucs_max(col_width[1],
                           (int)ucs_string_buffer_length(&select_param_strb) -
                                   col_width[2]);

    /* Print header */
    hdr_col_width[0] = col_width[0];
    hdr_col_width[1] = col_width[1] + 3 + col_width[2];
    ucp_proto_table_row_separator(strb, hdr_col_width, 2);
    ucs_string_buffer_appendf(strb, "| %*s | %-*s |\n", hdr_col_width[0],
                              ucs_string_buffer_cstr(&ep_cfg_strb),
                              hdr_col_width[1],
                              ucs_string_buffer_cstr(&select_param_strb));

    /* Print contents */
    ucp_proto_table_row_separator(strb, col_width, 3);
    ucs_array_for_each(row_elem, &table) {
        ucs_string_buffer_appendf(strb, info_row_fmt, col_width[0],
                                  row_elem->range_str, col_width[1],
                                  row_elem->desc, col_width[2],
                                  row_elem->config);
    }
    ucp_proto_table_row_separator(strb, col_width, 3);

    ucs_array_cleanup_dynamic(&table);
}

void ucp_proto_select_info(ucp_worker_h worker,
                           ucp_worker_cfg_index_t ep_cfg_index,
                           ucp_worker_cfg_index_t rkey_cfg_index,
                           const ucp_proto_select_t *proto_select,
                           ucs_string_buffer_t *strb)
{
    ucp_proto_select_elem_t select_elem;
    ucp_proto_select_key_t key;

    kh_foreach(&proto_select->hash, key.u64, select_elem,
               ucp_proto_select_elem_info(worker, ep_cfg_index, rkey_cfg_index,
                                          &key.param, &select_elem, strb);
               ucs_string_buffer_appendf(strb, "\n"))
}

void ucp_proto_select_dump_short(const ucp_proto_select_short_t *select_short,
                                 const char *name, ucs_string_buffer_t *strb)
{
    if (select_short->lane == UCP_NULL_LANE) {
        return;
    }

    ucs_string_buffer_appendf(strb, "\n%s: ", name);

    if (select_short->max_length_unknown_mem >= 0) {
        ucs_string_buffer_appendf(strb, "<= %zd",
                                  select_short->max_length_unknown_mem);
    } else {
        ucs_string_buffer_appendf(strb, "<= %zd and host memory",
                                  select_short->max_length_host_mem);
    }

    ucs_string_buffer_appendf(strb, ", using lane %d rkey_index %d\n",
                              select_short->lane, select_short->rkey_index);
}

static int ucp_proto_op_is_fetch(ucp_operation_id_t op_id)
{
    return (op_id == UCP_OP_ID_GET) || (op_id == UCP_OP_ID_RNDV_RECV);
}

void ucp_proto_select_param_str(const ucp_proto_select_param_t *select_param,
                                const char **operation_names,
                                ucs_string_buffer_t *strb)
{
    static const uint64_t op_attr_bits = UCP_OP_ATTR_FLAG_FAST_CMPL |
                                         UCP_OP_ATTR_FLAG_MULTI_SEND;
    static const char *op_attr_names[] = {
        [ucs_ilog2(UCP_OP_ATTR_FLAG_FAST_CMPL)]  = "fast-completion",
        [ucs_ilog2(UCP_OP_ATTR_FLAG_MULTI_SEND)] = "multi",
    };
    const char *sysdev_name;
    uint32_t op_attr_mask;

    ucs_string_buffer_appendf(strb, "%s", operation_names[select_param->op_id]);

    op_attr_mask = ucp_proto_select_op_attr_from_flags(select_param->op_flags);
    ucs_string_buffer_appendf(strb, "(");
    if (op_attr_mask & op_attr_bits) {
        ucs_string_buffer_append_flags(strb, op_attr_mask & op_attr_bits,
                                       op_attr_names);
    }
    ucs_string_buffer_appendf(strb, ")");

    if (ucp_proto_op_is_fetch(select_param->op_id)) {
        ucs_string_buffer_appendf(strb, " into ");
    } else {
        ucs_string_buffer_appendf(strb, " from ");
    }

    if (select_param->dt_class != UCP_DATATYPE_CONTIG) {
        ucs_string_buffer_appendf(
                strb, "%s", ucp_datatype_class_names[select_param->dt_class]);
        if (select_param->sg_count > 1) {
            ucs_string_buffer_appendf(strb, "[%d]", select_param->sg_count);
        }
        ucs_string_buffer_appendf(strb, " ");
    }

    ucs_string_buffer_appendf(strb, "%s",
                              ucs_memory_type_names[select_param->mem_type]);

    if (select_param->sys_dev == UCS_SYS_DEVICE_ID_UNKNOWN) {
        ucs_string_buffer_appendf(strb, " memory");
    } else {
        sysdev_name = ucs_topo_sys_device_get_name(select_param->sys_dev);
        ucs_string_buffer_appendf(strb, "/%s", sysdev_name);
    }
}

void ucp_proto_config_info_str(ucp_worker_h worker,
                               const ucp_proto_config_t *proto_config,
                               size_t msg_length, ucs_string_buffer_t *strb)
{
    const ucp_proto_select_elem_t *select_elem;
    ucp_worker_cfg_index_t new_key_cfg_index;
    const ucp_proto_select_range_t *range;
    ucp_proto_query_attr_t proto_attr;
    ucp_proto_select_t *proto_select;
    double bandwidth;

    ucs_assert(worker->context->config.ext.proto_enable);

    /* Print selection parameters */
    ucp_proto_select_param_str(&proto_config->select_param, ucp_operation_names,
                               strb);

    /* Print protocol description and configuration */
    ucp_proto_config_query(worker, proto_config, msg_length, &proto_attr);
    ucs_string_buffer_appendf(strb, " length %zu %s %s", msg_length,
                              proto_attr.desc, proto_attr.config);
    ucs_string_buffer_rtrim(strb, NULL);

    /* Find protocol selection root */
    proto_select = ucp_proto_select_get(worker, proto_config->ep_cfg_index,
                                        proto_config->rkey_cfg_index,
                                        &new_key_cfg_index);
    if (proto_select == NULL) {
        return;
    }

    /* Emulate protocol selection process */
    ucs_assert(new_key_cfg_index == proto_config->rkey_cfg_index);
    select_elem = ucp_proto_select_lookup_slow(worker, proto_select,
                                               proto_config->ep_cfg_index,
                                               proto_config->rkey_cfg_index,
                                               &proto_config->select_param);
    if (select_elem == NULL) {
        return;
    }

    /* Find the relevant performance range */
    range     = ucp_proto_perf_range_search(select_elem, msg_length);
    bandwidth = ucp_proto_select_calc_bandwidth(&proto_config->select_param,
                                                &range->super, msg_length);
    ucs_string_buffer_appendf(strb, " %.1f MB/s %.2f us", bandwidth / UCS_MBYTE,
                              msg_length / bandwidth * UCS_USEC_PER_SEC);
}

void ucp_proto_select_info_str(ucp_worker_h worker,
                               ucp_worker_cfg_index_t rkey_cfg_index,
                               const ucp_proto_select_param_t *select_param,
                               const char **operation_names,
                               ucs_string_buffer_t *strb)
{
    ucp_proto_select_param_str(select_param, operation_names, strb);

    if (rkey_cfg_index == UCP_WORKER_CFG_INDEX_NULL) {
        return;
    }

    if (ucp_proto_op_is_fetch(select_param->op_id)) {
        ucs_string_buffer_appendf(strb, " from ");
    } else {
        ucs_string_buffer_appendf(strb, " to ");
    }

    ucp_rkey_config_dump_brief(&worker->rkey_config[rkey_cfg_index].key, strb);
}

void ucp_proto_select_elem_trace(ucp_worker_h worker,
                                 ucp_worker_cfg_index_t ep_cfg_index,
                                 ucp_worker_cfg_index_t rkey_cfg_index,
                                 const ucp_proto_select_param_t *select_param,
                                 ucp_proto_select_elem_t *select_elem)
{
    ucs_string_buffer_t strb;
    char *line;

    if (!worker->context->config.ext.proto_info ||
        (select_param->op_flags & UCP_PROTO_SELECT_OP_FLAG_INTERNAL)) {
        return;
    }

    /* Print human-readable protocol selection table to the log */
    ucs_string_buffer_init(&strb);
    ucp_proto_select_elem_info(worker, ep_cfg_index, rkey_cfg_index,
                               select_param, select_elem, &strb);
    ucs_string_buffer_for_each_token(line, &strb, "\n") {
        ucs_log_print_compact(line);
    }
    ucs_string_buffer_cleanup(&strb);
}
