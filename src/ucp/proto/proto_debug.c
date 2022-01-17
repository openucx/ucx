/*
 * Copyright (C) 2022, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "proto_debug.h"
#include "proto_select.inl"


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
    ucp_proto_caps_t *proto_caps = init_params->caps;
    const UCS_V_UNUSED ucs_linear_func_t *perf;
    size_t UCS_V_UNUSED range_start, range_end;
    ucs_string_buffer_t config_strb;
    int UCS_V_UNUSED range_index;
    char min_length_str[64];
    char thresh_str[64];

    ucs_string_buffer_init(&config_strb);
    ucp_proto_id_call(proto_id, config_str, proto_caps->min_length, SIZE_MAX,
                      init_params->priv, &config_strb);
    ucs_trace("initialized protocol %s min_length %s cfg_thresh %s %s",
              init_params->proto_name,
              ucs_memunits_to_str(proto_caps->min_length, min_length_str,
                                  sizeof(min_length_str)),
              ucs_memunits_to_str(proto_caps->cfg_thresh, thresh_str,
                                  sizeof(thresh_str)),
              ucs_string_buffer_cstr(&config_strb));
    ucs_string_buffer_cleanup(&config_strb);

    ucs_log_indent(1);
    range_start = 0;
    for (range_index = 0; range_index < proto_caps->num_ranges; ++range_index) {
        range_start = ucs_max(range_start, proto_caps->min_length);
        range_end   = proto_caps->ranges[range_index].max_length;
        if (range_end > range_start) {
            perf = proto_caps->ranges[range_index].perf;
            ucs_trace("range[%d] %s %s" UCP_PROTO_PERF_FUNC_TYPES_FMT,
                      range_index, proto_caps->ranges[range_index].name,
                      ucs_memunits_range_str(range_start, range_end, thresh_str,
                                             sizeof(thresh_str)),
                      UCP_PROTO_PERF_FUNC_TYPES_ARG(perf));
        }
        range_start = range_end + 1;
    }
    ucs_log_indent(-1);
}

void ucp_proto_select_dump_thresholds(
        const ucp_proto_select_elem_t *select_elem, ucs_string_buffer_t *strb)
{
    static const char *proto_info_fmt = "    %-18s %-18s %s\n";
    const ucp_proto_threshold_elem_t *thresh_elem;
    ucs_string_buffer_t proto_config_strb;
    size_t range_start, range_end;
    char range_str[128];

    range_start = 0;
    thresh_elem = select_elem->thresholds;
    ucs_string_buffer_appendf(strb, proto_info_fmt, "SIZE", "PROTOCOL",
                              "CONFIGURATION");
    do {
        ucs_string_buffer_init(&proto_config_strb);

        range_end = thresh_elem->max_msg_length;
        thresh_elem->proto_config.proto->config_str(
                range_start, range_end, thresh_elem->proto_config.priv,
                &proto_config_strb);

        ucs_memunits_range_str(range_start, range_end, range_str,
                               sizeof(range_str));

        ucs_string_buffer_appendf(strb, proto_info_fmt, range_str,
                                  thresh_elem->proto_config.proto->name,
                                  ucs_string_buffer_cstr(&proto_config_strb));

        ucs_string_buffer_cleanup(&proto_config_strb);

        range_start = range_end + 1;
        ++thresh_elem;
    } while (range_end != SIZE_MAX);
}

static void
ucp_proto_select_dump_perf(const ucp_proto_select_elem_t *select_elem,
                           ucp_proto_perf_type_t perf_type,
                           ucs_string_buffer_t *strb)
{
    static const char *proto_info_fmt = "    %-16s %-20s %s\n";
    const ucp_proto_select_range_t *range_elem;
    size_t range_start, range_end;
    char range_str[128];
    char time_str[64];
    char bw_str[64];

    range_start = 0;
    range_elem  = select_elem->perf_ranges;
    ucs_string_buffer_appendf(strb, proto_info_fmt, "SIZE", "TIME (nsec)",
                              "BANDWIDTH (MiB/s)");
    do {
        range_end = range_elem->super.max_length;

        ucp_proto_select_perf_str(&range_elem->super.perf[perf_type], time_str,
                                  sizeof(time_str), bw_str, sizeof(bw_str));
        ucs_memunits_range_str(range_start, range_end, range_str,
                               sizeof(range_str));

        ucs_string_buffer_appendf(strb, proto_info_fmt, range_str, time_str,
                                  bw_str);

        range_start = range_end + 1;
        ++range_elem;
    } while (range_end != SIZE_MAX);
}

static void
ucp_proto_select_elem_dump(ucp_worker_h worker,
                           ucp_worker_cfg_index_t ep_cfg_index,
                           ucp_worker_cfg_index_t rkey_cfg_index,
                           const ucp_proto_select_param_t *select_param,
                           const ucp_proto_select_elem_t *select_elem,
                           ucs_string_buffer_t *strb)
{
    UCS_STRING_BUFFER_ONSTACK(sel_param_strb, UCP_PROTO_SELECT_PARAM_STR_MAX);
    size_t i;

    ucp_proto_select_param_str(select_param, &sel_param_strb);

    ucs_string_buffer_appendf(strb, "  %s\n  ",
                              ucs_string_buffer_cstr(&sel_param_strb));
    for (i = 0; i < ucs_string_buffer_length(&sel_param_strb); ++i) {
        ucs_string_buffer_appendf(strb, "=");
    }
    ucs_string_buffer_appendf(strb, "\n");

    ucs_string_buffer_appendf(strb, "\n  Selected protocols:\n");
    ucp_proto_select_dump_thresholds(select_elem, strb);

    ucs_string_buffer_appendf(strb, "\n  Performance estimation:\n");
    ucp_proto_select_dump_perf(select_elem,
                               ucp_proto_select_param_perf_type(select_param),
                               strb);
}

void ucp_proto_select_dump(ucp_worker_h worker,
                           ucp_worker_cfg_index_t ep_cfg_index,
                           ucp_worker_cfg_index_t rkey_cfg_index,
                           const ucp_proto_select_t *proto_select,
                           ucs_string_buffer_t *strb)
{
    ucp_proto_select_elem_t select_elem;
    ucp_proto_select_key_t key;
    char info[256];

    ucp_worker_print_used_tls(&worker->ep_config[ep_cfg_index].key,
                              worker->context, ep_cfg_index, info,
                              sizeof(info));
    ucs_string_buffer_appendf(strb, "\nProtocol selection for %s", info);

    if (rkey_cfg_index != UCP_WORKER_CFG_INDEX_NULL) {
        ucs_string_buffer_appendf(strb, "rkey_cfg[%d]: ", rkey_cfg_index);
        ucp_rkey_config_dump_brief(&worker->rkey_config[rkey_cfg_index].key,
                                   strb);
    }
    ucs_string_buffer_appendf(strb, "\n\n");

    if (kh_size(&proto_select->hash) == 0) {
        ucs_string_buffer_appendf(strb, "   (No elements)\n");
        return;
    }

    kh_foreach(&proto_select->hash, key.u64, select_elem,
               ucp_proto_select_elem_dump(worker, ep_cfg_index, rkey_cfg_index,
                                          &key.param, &select_elem, strb))
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

void ucp_proto_select_param_str(const ucp_proto_select_param_t *select_param,
                                ucs_string_buffer_t *strb)
{
    const char *sysdev_name;
    uint32_t op_attr_mask;

    op_attr_mask = ucp_proto_select_op_attr_from_flags(select_param->op_flags);
    ucs_string_buffer_appendf(strb, "%s(",
                              ucp_operation_names[select_param->op_id]);

    ucs_string_buffer_appendf(strb, "%s",
                              ucp_datatype_class_names[select_param->dt_class]);

    if (select_param->sg_count > 1) {
        ucs_string_buffer_appendf(strb, "[%d]", select_param->sg_count);
    }

    if (select_param->mem_type != UCS_MEMORY_TYPE_HOST) {
        ucs_string_buffer_appendf(
                strb, ", %s", ucs_memory_type_names[select_param->mem_type]);
    }

    if (select_param->sys_dev != UCS_SYS_DEVICE_ID_UNKNOWN) {
        sysdev_name = ucs_topo_sys_device_get_name(select_param->sys_dev);
        ucs_string_buffer_appendf(strb, ", %s", sysdev_name);
    }

    if (op_attr_mask & UCP_OP_ATTR_FLAG_FAST_CMPL) {
        ucs_string_buffer_appendf(strb, ", fast-completion");
    }

    if (op_attr_mask & UCP_OP_ATTR_FLAG_MULTI_SEND) {
        ucs_string_buffer_appendf(strb, ", multi");
    }

    ucs_string_buffer_rtrim(strb, ",");
    ucs_string_buffer_appendf(strb, ")");
}

void ucp_proto_threshold_elem_str(const ucp_proto_threshold_elem_t *thresh_elem,
                                  size_t min_length, size_t max_length,
                                  ucs_string_buffer_t *strb)
{
    size_t range_start, range_end;
    const ucp_proto_t *proto;
    char str[64];

    range_start = 0;
    do {
        range_end = thresh_elem->max_msg_length;

        /* Print only protocols within the range provided by {min,max}_length */
        if ((range_end >= min_length) && (range_start <= max_length)) {
            proto = thresh_elem->proto_config.proto;
            ucs_string_buffer_appendf(strb, "%s(", proto->name);
            proto->config_str(ucs_max(range_start, min_length),
                              ucs_min(range_end, max_length),
                              thresh_elem->proto_config.priv, strb);
            ucs_string_buffer_appendf(strb, ")");

            if (range_end < max_length) {
                ucs_memunits_to_str(thresh_elem->max_msg_length, str,
                                    sizeof(str));
                ucs_string_buffer_appendf(strb, "<=%s<", str);
            }
        }

        ++thresh_elem;
        range_start = range_end + 1;
    } while (range_end < max_length);

    ucs_string_buffer_rtrim(strb, "<");
}

void ucp_proto_select_config_str(ucp_worker_h worker,
                                 const ucp_proto_config_t *proto_config,
                                 size_t msg_length, ucs_string_buffer_t *strb)
{
    const ucp_proto_select_elem_t *select_elem;
    ucp_worker_cfg_index_t new_key_cfg_index;
    const ucp_proto_select_range_t *range;
    ucp_proto_select_t *proto_select;
    ucp_proto_perf_type_t perf_type;
    double send_time;

    ucs_assert(worker->context->config.ext.proto_enable);

    /* Print selection parameters */
    ucp_proto_select_param_str(&proto_config->select_param, strb);
    ucs_string_buffer_appendf(strb, ": %s ", proto_config->proto->name);
    proto_config->proto->config_str(msg_length, msg_length, proto_config->priv,
                                    strb);

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
    range = select_elem->perf_ranges;
    while (range->super.max_length < msg_length) {
        ++range;
    }

    perf_type = ucp_proto_select_param_perf_type(&proto_config->select_param);
    send_time = ucs_linear_func_apply(range->super.perf[perf_type], msg_length);
    ucs_string_buffer_appendf(strb, "  %.2f MBs / %.2f us",
                              (msg_length / send_time) / UCS_MBYTE,
                              send_time * UCS_USEC_PER_SEC);
}
