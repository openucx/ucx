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

#include <ucp/am/ucp_am.inl>
#include <ucp/rndv/proto_rndv.h>
#include <ucs/arch/atomic.h>
#include <fnmatch.h>
#include <ctype.h>


/* Performance node data entry */
typedef struct {
    const char        *name;
    ucs_linear_func_t value;
} ucp_proto_perf_node_data_t;

/*
 * Performance estimation for a range of message sizes.
 * Defined in C file to prevent direct access to the structure fields.
 */
struct ucp_proto_perf_node {
    /* Type of the range */
    ucp_proto_perf_node_type_t                    type;

    /* Name of the range */
    char                                          name[UCP_PROTO_DESC_STR_MAX];

    /* Description of the range */
    char                                          desc[UCP_PROTO_DESC_STR_MAX];

    /* Number of references in the performance tree defined by 'children' */
    unsigned                                      refcount;

    /* Array of child performance node pointers */
    ucs_array_s(unsigned, ucp_proto_perf_node_t*) children;

    union {
        /*
         * Index of selected child node in the 'children' array.
         * Used when type == UCP_PROTO_PERF_NODE_TYPE_SELECT
         */
        unsigned                                          selected_child;

        /*
         * Array of performance data entries.
         * Used when type == UCP_PROTO_PERF_NODE_TYPE_DATA
         */
        ucs_array_s(unsigned, ucp_proto_perf_node_data_t) data;
    };
};

/* Protocol information table */
typedef struct {
    char range_str[32];
    char desc[UCP_PROTO_DESC_STR_MAX];
    char config[UCP_PROTO_CONFIG_STR_MAX];
} ucp_proto_info_row_t;
UCS_ARRAY_DECLARE_TYPE(ucp_proto_info_table_t, unsigned, ucp_proto_info_row_t);


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

void ucp_proto_select_init_trace_perf(const ucp_proto_init_params_t *init_params,
                                      const ucp_proto_perf_t *perf,
                                      const void *priv)
{
    ucp_proto_query_params_t query_params = {
        .proto         = ucp_protocols[init_params->proto_id],
        .priv          = priv,
        .worker        = init_params->worker,
        .select_param  = init_params->select_param,
        .ep_config_key = init_params->ep_config_key
    };
    UCS_STRING_BUFFER_ONSTACK(seg_strb, 128);
    const ucp_proto_perf_segment_t *seg;
    ucp_proto_query_attr_t query_attr;
    size_t seg_start, seg_end, range_start, range_end;
    char range_str[64];

    if (!ucs_log_is_enabled(UCS_LOG_LEVEL_TRACE)) {
        return;
    }

    range_end = -1;
    do {
        range_start             = range_end + 1;
        query_params.msg_length = range_start;
        ucp_proto_id_call(init_params->proto_id, query, &query_params,
                          &query_attr);

        range_end = query_attr.max_msg_length;
        ucp_proto_perf_segment_foreach_range(seg, seg_start, seg_end, perf,
                                             range_start, range_end) {
            ucs_string_buffer_reset(&seg_strb);
            ucp_proto_perf_segment_str(seg, &seg_strb);
            ucs_trace("%s: %s %s %s", ucs_string_buffer_cstr(&seg_strb),
                      ucs_memunits_range_str(seg_start, seg_end, range_str,
                                             sizeof(range_str)),
                      query_attr.desc, query_attr.config);
        }
    } while (range_end < SIZE_MAX);
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
    ucp_ep_config_name(worker, ep_cfg_index, ep_cfg_strb);

    /* Operation name and attributes */
    ucp_proto_select_info_str(worker, rkey_cfg_index, select_param,
                              operation_names, select_param_strb);
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

static int ucp_proto_debug_is_info_enabled(ucp_context_h context,
                                           const char *select_param_str)
{
    const char *proto_info_config = context->config.ext.proto_info;
    int bool_value;

    if (ucs_config_sscanf_bool(proto_info_config, &bool_value, NULL)) {
        return bool_value;
    }

    return fnmatch(proto_info_config, select_param_str, FNM_CASEFOLD) == 0;
}

static void
ucp_proto_select_elem_info(ucp_worker_h worker,
                           ucp_worker_cfg_index_t ep_cfg_index,
                           ucp_worker_cfg_index_t rkey_cfg_index,
                           const ucp_proto_select_param_t *select_param,
                           ucp_proto_select_elem_t *select_elem, int show_all,
                           ucs_string_buffer_t *strb)
{
    UCS_STRING_BUFFER_ONSTACK(ep_cfg_strb, UCP_PROTO_CONFIG_STR_MAX);
    UCS_STRING_BUFFER_ONSTACK(sel_param_strb, UCP_PROTO_CONFIG_STR_MAX);
    static const char *info_row_fmt = "| %*s | %-*s | %-*s |\n";
    ucp_proto_info_table_t table;
    int hdr_col_width[2], col_width[3];
    ucp_proto_query_attr_t proto_attr;
    ucp_proto_info_row_t *row_elem;
    size_t range_start, range_end;
    int proto_valid;

    ucp_proto_select_param_dump(worker, ep_cfg_index, rkey_cfg_index,
                                select_param, ucp_operation_descs, &ep_cfg_strb,
                                &sel_param_strb);
    if (!show_all &&
        !ucp_proto_debug_is_info_enabled(
                worker->context, ucs_string_buffer_cstr(&sel_param_strb))) {
        return;
    }

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

        row_elem = ucs_array_append(&table, break);

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
                           (int)ucs_string_buffer_length(&sel_param_strb) -
                                   col_width[2]);

    /* Print header */
    hdr_col_width[0] = col_width[0];
    hdr_col_width[1] = col_width[1] + 3 + col_width[2];
    ucp_proto_table_row_separator(strb, hdr_col_width, 2);
    ucs_string_buffer_appendf(strb, "| %*s | %-*s |\n", hdr_col_width[0],
                              ucs_string_buffer_cstr(&ep_cfg_strb),
                              hdr_col_width[1],
                              ucs_string_buffer_cstr(&sel_param_strb));

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
                           const ucp_proto_select_t *proto_select, int show_all,
                           ucs_string_buffer_t *strb)
{
    ucp_proto_select_elem_t select_elem;
    ucp_proto_select_key_t key;

    kh_foreach(proto_select->hash, key.u64, select_elem,
               ucp_proto_select_elem_info(worker, ep_cfg_index, rkey_cfg_index,
                                          &key.param, &select_elem, show_all,
                                          strb);
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

static int
ucp_proto_select_is_fetch_op(const ucp_proto_select_param_t *select_param)
{
    return ucp_proto_select_check_op(select_param,
                                     UCS_BIT(UCP_OP_ID_GET) |
                                     UCS_BIT(UCP_OP_ID_RNDV_RECV) |
                                     UCS_BIT(UCP_OP_ID_AMO_FETCH));
}

static int
ucp_proto_select_is_rndv_op(const ucp_proto_select_param_t *select_param)
{
    return ucp_proto_select_check_op(select_param, UCP_PROTO_RNDV_OP_ID_MASK);
}

static int
ucp_proto_select_is_am_op(const ucp_proto_select_param_t *select_param)
{
    return ucp_proto_select_check_op(select_param, UCP_PROTO_AM_OP_ID_MASK);
}

static int
ucp_proto_select_is_atomic_op(const ucp_proto_select_param_t *select_param)
{
    return ucp_proto_select_check_op(select_param,
                                     UCS_BIT(UCP_OP_ID_AMO_POST) |
                                     UCS_BIT(UCP_OP_ID_AMO_FETCH) |
                                     UCS_BIT(UCP_OP_ID_AMO_CSWAP));
}
static void ucp_proto_debug_mem_info_str(ucs_string_buffer_t *strb,
                                         ucs_memory_type_t mem_type,
                                         ucs_sys_device_t sys_dev)
{
    const char *sysdev_name;

    ucs_string_buffer_appendf(strb, "%s", ucs_memory_type_names[mem_type]);

    if (sys_dev == UCS_SYS_DEVICE_ID_UNKNOWN) {
        ucs_string_buffer_appendf(strb, " memory");
    } else {
        sysdev_name = ucs_topo_sys_device_get_name(sys_dev);
        ucs_string_buffer_appendf(strb, "/%s", sysdev_name);
    }
}

void ucp_proto_select_param_str(const ucp_proto_select_param_t *select_param,
                                const char **operation_names,
                                ucs_string_buffer_t *strb)
{
    static const char *op_attr_names[]   = {
        [ucs_ilog2(UCP_OP_ATTR_FLAG_FAST_CMPL)]  = "fast-completion",
        [ucs_ilog2(UCP_OP_ATTR_FLAG_MULTI_SEND)] = "multi",
    };
    static const char *rndv_flag_names[] = {
        [ucs_ilog2(UCP_PROTO_SELECT_OP_FLAG_PPLN_FRAG)] = "frag"
    };
    static const char *am_flag_names[]   = {
        [ucs_ilog2(UCP_PROTO_SELECT_OP_FLAG_AM_EAGER)] = "egr",
        [ucs_ilog2(UCP_PROTO_SELECT_OP_FLAG_AM_RNDV)]  = "rndv"
    };
    uint32_t op_attr_mask, op_flags;

    ucs_string_buffer_appendf(
            strb, "%s", operation_names[ucp_proto_select_op_id(select_param)]);

    op_attr_mask = ucp_proto_select_op_attr_unpack(select_param->op_attr);
    op_flags     = ucp_proto_select_op_flags(select_param);

    if (op_attr_mask || op_flags) {
        ucs_string_buffer_appendf(strb, "(");
        if (op_attr_mask) {
            ucs_string_buffer_append_flags(strb, op_attr_mask, op_attr_names);
            ucs_string_buffer_appendf(strb, ",");
        }
        if (op_flags) {
            if (ucp_proto_select_is_rndv_op(select_param)) {
                ucs_string_buffer_append_flags(strb, op_flags, rndv_flag_names);
            } else if (ucp_proto_select_is_am_op(select_param)) {
                ucs_string_buffer_append_flags(strb, op_flags, am_flag_names);
            }
        }
        ucs_string_buffer_rtrim(strb, ",");
        ucs_string_buffer_appendf(strb, ")");
    }

    if (ucp_proto_select_op_id(select_param) == UCP_OP_ID_AMO_POST) {
        /* No need to print reply buffer info for AMO post */
        return;
    }

    if (ucp_proto_select_is_fetch_op(select_param)) {
        ucs_string_buffer_appendf(strb, " into ");
    } else if (ucp_proto_select_op_id(select_param) == UCP_OP_ID_AMO_CSWAP) {
        ucs_string_buffer_appendf(strb, " of ");
    } else {
        ucs_string_buffer_appendf(strb, " from ");
    }

    if (ucp_proto_select_is_atomic_op(select_param)) {
        /* Atomic fetch/cswap prints the reply buffer info */
        ucp_proto_debug_mem_info_str(strb, select_param->op.reply.mem_type,
                                     select_param->op.reply.sys_dev);
        return;
    }

    if (select_param->dt_class != UCP_DATATYPE_CONTIG) {
        ucs_string_buffer_appendf(
                strb, "%s", ucp_datatype_class_names[select_param->dt_class]);
        if (select_param->sg_count > 1) {
            ucs_string_buffer_appendf(strb, "[%d]", select_param->sg_count);
        }
        ucs_string_buffer_appendf(strb, " ");
    }

    ucp_proto_debug_mem_info_str(strb, select_param->mem_type,
                                 select_param->sys_dev);
}

void ucp_proto_config_info_str(ucp_worker_h worker,
                               const ucp_proto_config_t *proto_config,
                               size_t msg_length, ucs_string_buffer_t *strb)
{
    const ucp_proto_flat_perf_range_t *range;
    ucp_proto_query_attr_t proto_attr;
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

    /* Find the relevant performance range */
    range = ucp_proto_flat_perf_find_lb(proto_config->init_elem->flat_perf,
                                        msg_length);
    if ((range == NULL) || (range->start > msg_length)) {
        ucs_string_buffer_appendf(strb, " - not available");
        return;
    }

    bandwidth = msg_length / ucs_linear_func_apply(range->value, msg_length);
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

    if (rkey_cfg_index != UCP_WORKER_CFG_INDEX_NULL) {
        if (ucp_proto_select_is_fetch_op(select_param)) {
            ucs_string_buffer_appendf(strb, " from ");
        } else if (ucp_proto_select_op_id(select_param) ==
                   UCP_OP_ID_AMO_CSWAP) {
            ucs_string_buffer_appendf(strb, " with ");
        } else {
            ucs_string_buffer_appendf(strb, " to ");
        }

        ucp_rkey_config_dump_brief(&worker->rkey_config[rkey_cfg_index].key,
                                   strb);
    }

    if (ucp_proto_select_is_atomic_op(select_param)) {
        ucs_string_buffer_appendf(strb, ", arg in ");
        ucp_proto_debug_mem_info_str(strb, select_param->mem_type,
                                     select_param->sys_dev);
    }
}

ucp_proto_perf_node_t *ucp_proto_perf_node_new(ucp_proto_perf_node_type_t type,
                                               unsigned selected_child,
                                               const char *name,
                                               const char *desc_fmt, va_list ap)
{
    ucp_proto_perf_node_t *perf_node;

    perf_node = ucs_malloc(sizeof(*perf_node), "ucp_proto_perf_node");
    if (perf_node == NULL) {
        return NULL;
    }

    perf_node->type     = type;
    perf_node->refcount = 1;
    ucs_array_init_dynamic(&perf_node->children);

    ucs_assert(name != NULL);
    ucs_strncpy_safe(perf_node->name, name, sizeof(perf_node->name));
    ucs_vsnprintf_safe(perf_node->desc, sizeof(perf_node->desc), desc_fmt, ap);

    if (type == UCP_PROTO_PERF_NODE_TYPE_DATA) {
        ucs_array_init_dynamic(&perf_node->data);
    } else if (type == UCP_PROTO_PERF_NODE_TYPE_SELECT) {
        perf_node->selected_child = selected_child;
    }
    return perf_node;
}

static void ucp_proto_perf_node_free(ucp_proto_perf_node_t *perf_node)
{
    ucp_proto_perf_node_t **child_elem;

    /* Delete children recursively */
    ucs_array_for_each(child_elem, &perf_node->children) {
        ucp_proto_perf_node_deref(child_elem);
    }
    ucs_array_cleanup_dynamic(&perf_node->children);

    if (perf_node->type == UCP_PROTO_PERF_NODE_TYPE_DATA) {
        ucs_array_cleanup_dynamic(&perf_node->data);
    }

    ucs_free(perf_node);
}

#define UCP_PROTO_PERF_NODE_NEW(_type, _selected_child, _name, _desc_fmt) \
    ({ \
        ucp_proto_perf_node_t *__perf_node; \
        va_list __ap; \
        \
        va_start(__ap, _desc_fmt); \
        __perf_node = ucp_proto_perf_node_new(UCP_PROTO_PERF_NODE_TYPE_##_type, \
                                              _selected_child, _name, \
                                              _desc_fmt, __ap); \
        va_end(__ap); \
        \
        if (__perf_node == NULL) { \
            return NULL; \
        } \
        \
        __perf_node; \
    })

ucp_proto_perf_node_t *
ucp_proto_perf_node_new_data(const char *name, const char *desc_fmt, ...)
{
    return UCP_PROTO_PERF_NODE_NEW(DATA, 0, name, desc_fmt);
}

ucp_proto_perf_node_t *ucp_proto_perf_node_new_select(const char *name,
                                                      unsigned selected_child,
                                                      const char *desc_fmt, ...)
{
    return UCP_PROTO_PERF_NODE_NEW(SELECT, selected_child, name, desc_fmt);
}

void ucp_proto_perf_node_ref(ucp_proto_perf_node_t *perf_node)
{
    if (perf_node != NULL) {
        ucs_assert(perf_node->refcount != UINT_MAX);
        ++perf_node->refcount;
    }
}

void ucp_proto_perf_node_deref(ucp_proto_perf_node_t **perf_node_p)
{
    ucp_proto_perf_node_t *perf_node = *perf_node_p;

    if (perf_node == NULL) {
        return;
    }

    ucs_assertv(perf_node->refcount > 0, "perf_node=%p name='%s' desc='%s'",
                perf_node, perf_node->name, perf_node->desc);
    --perf_node->refcount;
    if (perf_node->refcount == 0) {
        ucp_proto_perf_node_free(perf_node);
    }

    *perf_node_p = NULL;
}

static void
ucp_proto_perf_node_append_child(ucp_proto_perf_node_t *perf_node,
                                 ucp_proto_perf_node_t *child_perf_node)
{
    ucs_array_append(&perf_node->children,
                     ucs_diag("failed to add perf node child");
                     return );
    *ucs_array_last(&perf_node->children) = child_perf_node;
}

ucp_proto_perf_node_t *
ucp_proto_perf_node_dup(const ucp_proto_perf_node_t *perf_node)
{
    ucp_proto_perf_node_t *dup_perf_node = NULL;
    ucp_proto_perf_node_t **child_elem;
    ucp_proto_perf_node_data_t *data;

    if (perf_node == NULL) {
        return NULL;
    }

    if (perf_node->type == UCP_PROTO_PERF_NODE_TYPE_DATA) {
        dup_perf_node = ucp_proto_perf_node_new_data(perf_node->name, "%s",
                                                     perf_node->desc);
    } else if (perf_node->type == UCP_PROTO_PERF_NODE_TYPE_SELECT) {
        dup_perf_node = ucp_proto_perf_node_new_select(perf_node->name,
                                                       perf_node->selected_child,
                                                       "%s", perf_node->desc);
    }
    if (dup_perf_node == NULL) {
        return NULL;
    }

    ucs_array_for_each(child_elem, &perf_node->children) {
        ucp_proto_perf_node_add_child(dup_perf_node, *child_elem);
    }

    if (perf_node->type == UCP_PROTO_PERF_NODE_TYPE_DATA) {
        ucs_array_for_each(data, &perf_node->data) {
            ucp_proto_perf_node_add_data(dup_perf_node, data->name,
                                         data->value);
        }
    }

    return dup_perf_node;
}

void ucp_proto_perf_node_own_child(ucp_proto_perf_node_t *perf_node,
                                   ucp_proto_perf_node_t **child_perf_node_p)
{
    if (*child_perf_node_p == NULL) {
        return;
    } else if (perf_node == NULL) {
        ucp_proto_perf_node_deref(child_perf_node_p);
        return;
    }

    ucp_proto_perf_node_append_child(perf_node, *child_perf_node_p);
}

void ucp_proto_perf_node_add_child(ucp_proto_perf_node_t *perf_node,
                                   ucp_proto_perf_node_t *child_perf_node)
{
    if ((perf_node == NULL) || (child_perf_node == NULL)) {
        return;
    }

    ucp_proto_perf_node_append_child(perf_node, child_perf_node);
    ucp_proto_perf_node_ref(child_perf_node);
}

ucp_proto_perf_node_t *
ucp_proto_perf_node_get_child(ucp_proto_perf_node_t *perf_node, unsigned n)
{
    if ((perf_node == NULL) || (n >= ucs_array_length(&perf_node->children))) {
        return NULL;
    }

    return ucs_array_elem(&perf_node->children, n);
}

void ucp_proto_perf_node_add_data(ucp_proto_perf_node_t *perf_node,
                                  const char *name,
                                  const ucs_linear_func_t value)
{
    ucp_proto_perf_node_data_t *data;

    if (perf_node == NULL) {
        return;
    }

    ucs_assert(perf_node->type == UCP_PROTO_PERF_NODE_TYPE_DATA);

    ucs_array_append(&perf_node->data, ucs_diag("failed to add perf node data");
                     return );
    data        = ucs_array_last(&perf_node->data);
    data->name  = name;
    data->value = value;
}

void ucp_proto_perf_node_update_data(ucp_proto_perf_node_t *perf_node,
                                     const char *name,
                                     const ucs_linear_func_t value)
{
    ucp_proto_perf_node_data_t *data;

    if (perf_node == NULL) {
        return;
    }

    ucs_array_for_each(data, &perf_node->data) {
        if (!strcmp(name, data->name)) {
            data->value = value;
            return;
        }
    }

    ucp_proto_perf_node_add_data(perf_node, name, value);
}

void ucp_proto_perf_node_add_scalar(ucp_proto_perf_node_t *perf_node,
                                    const char *name, double value)
{
    ucp_proto_perf_node_add_data(perf_node, name,
                                 ucs_linear_func_make(value, 0));
}

void ucp_proto_perf_node_add_bandwidth(ucp_proto_perf_node_t *perf_node,
                                       const char *name, double value)
{
    if (value > UCP_PROTO_PERF_EPSILON) {
        ucp_proto_perf_node_add_data(perf_node, name,
                                     ucs_linear_func_make(0, 1.0 / value));
    }
}

const char *ucp_proto_perf_node_name(ucp_proto_perf_node_t *perf_node)
{
    return (perf_node == NULL) ? "(null)" : perf_node->name;
}

const char *ucp_proto_perf_node_desc(ucp_proto_perf_node_t *perf_node)
{
    return (perf_node == NULL) ? "(null)" : perf_node->desc;
}

void ucp_proto_perf_node_replace(ucp_proto_perf_node_t **old_perf_node_p,
                                 ucp_proto_perf_node_t **new_perf_node_p)
{
    ucp_proto_perf_node_t **child_elem;

    if (*old_perf_node_p != NULL) {
        ucs_array_for_each(child_elem, &(*old_perf_node_p)->children) {
            ucp_proto_perf_node_add_child(*new_perf_node_p, *child_elem);
        }
    }

    ucp_proto_perf_node_deref(old_perf_node_p);
    *old_perf_node_p = *new_perf_node_p;
    *new_perf_node_p = NULL;
}

static void
ucp_proto_perf_graph_str_append_line_break(ucs_string_buffer_t *strb)
{
    ucs_string_buffer_appendf(strb, "<br align=\"left\"/>");
}

static void
ucp_proto_perf_graph_str_append_perf(const ucp_proto_perf_node_data_t *data,
                                     ucs_string_buffer_t *strb)
{
    double kbps;

    if (data->name[0] != '\0') {
        /* Show only non-empty name */
        ucs_string_buffer_appendf(strb, "%s ", data->name);
    }

    if (data->value.m > UCP_PROTO_PERF_EPSILON) {
        kbps = 1.0 / (data->value.m * UCS_KBYTE);
        if (kbps < 1e4) {
            ucs_string_buffer_appendf(strb, "%.0fKBs", kbps);
        } else if (kbps < 1e7) {
            ucs_string_buffer_appendf(strb, "%.0fMBs", kbps / 1024);
        } else {
            ucs_string_buffer_appendf(strb, "%.1fGBs", kbps / 1024 / 1024);
        }
        if (data->value.c > UCP_PROTO_PERF_EPSILON) {
            ucs_string_buffer_appendf(strb, "+");
        }
    }

    if (data->value.c > UCP_PROTO_PERF_EPSILON) {
        ucs_string_buffer_appendf(strb, "%0.2f&mu;s",
                                  data->value.c * UCS_USEC_PER_SEC);
    }
}

static void ucp_proto_perf_graph_str_append_link(int parent_id, int child_id,
                                                 int highlight,
                                                 ucs_string_buffer_t *strb)
{
    ucs_string_buffer_appendf(strb, "\tnode%d -> node%d", parent_id, child_id);
    if (highlight > 0) {
        ucs_string_buffer_appendf(strb, " [style=bold]");
    } else if (highlight < 0) {
        ucs_string_buffer_appendf(strb, " [style=dotted]");
    }
    ucs_string_buffer_appendf(strb, ";\n");
}

static inline khint32_t kh_ptr_hash_func(void *ptr)
{
    return kh_int64_hash_func((uintptr_t)ptr);
}

KHASH_INIT(ucp_proto_graph_node, ucp_proto_perf_node_t*, int, 1,
           kh_ptr_hash_func, kh_int64_hash_equal);

static void
ucp_proto_perf_graph_dump_recurs(ucp_proto_perf_node_t *perf_node,
                                 int parent_id,
                                 khash_t(ucp_proto_graph_node) *nodes_hash,
                                 int highlight, ucs_string_buffer_t *strb)
{
    UCS_STRING_BUFFER_ONSTACK(node_style, 64);
    ucp_proto_perf_node_t **child_elem;
    ucp_proto_perf_node_data_t *data;
    int khret, id, child_index;
    int child_highlight;
    const char *shape;
    khiter_t khiter;

    if (perf_node == NULL) {
        return;
    }

    khiter = kh_put(ucp_proto_graph_node, nodes_hash, perf_node, &khret);
    if (khret == UCS_KH_PUT_KEY_PRESENT) {
        id = kh_value(nodes_hash, khiter);
        ucp_proto_perf_graph_str_append_link(parent_id, id, highlight, strb);
        return;
    } else if (khret == UCS_KH_PUT_FAILED) {
        ucs_debug("failed to add node %p to performance tree hash", perf_node);
        return;
    }

    id = kh_size(nodes_hash); /* We never remove from the hash */
    kh_value(nodes_hash, khiter) = id;

    /* Connect to parent */
    ucp_proto_perf_graph_str_append_link(parent_id, id, highlight, strb);

    /* Header */
    switch (perf_node->type) {
    default:
    case UCP_PROTO_PERF_NODE_TYPE_DATA:
        shape = "note";
        break;
    case UCP_PROTO_PERF_NODE_TYPE_SELECT:
        shape = "oval";
        break;
    }

    /* Open node */
    ucs_string_buffer_appendf(strb, "\tnode%d [shape=%s ", id, shape);
    if (highlight > 0) {
        ucs_string_buffer_appendf(&node_style, "filled,");
        ucs_string_buffer_appendf(strb, "fillcolor=\"#FFE0E0\" ");
    }

    /* Style */
    ucs_string_buffer_rtrim(&node_style, ",");
    ucs_string_buffer_appendf(strb, "style=\"%s\" ",
                              ucs_string_buffer_cstr(&node_style));

    /* Start node label */
    ucs_string_buffer_appendf(strb, "label=<<b>");
    ucs_string_buffer_appendf(strb, "%s</b><br align=\"center\"/>",
                              perf_node->name);

    /* Description */
    if (!ucs_string_is_empty(perf_node->desc)) {
        ucs_string_buffer_appendf(
                strb, "<font face=\"calibri\" point-size=\"11\">%s"
                UCP_PROTO_PERF_NODE_NEW_LINE"</font>",
                perf_node->desc);
    }

    /* Data entries */
    if ((perf_node->type == UCP_PROTO_PERF_NODE_TYPE_DATA) &&
        !ucs_array_is_empty(&perf_node->data)) {
        ucs_string_buffer_appendf(strb,
                                  "<font face=\"calibri\" point-size=\"13\">");
        ucs_array_for_each(data, &perf_node->data) {
            if (ucs_linear_func_is_zero(data->value, UCP_PROTO_PERF_EPSILON)) {
                continue;
            }

            ucp_proto_perf_graph_str_append_line_break(strb);
            ucp_proto_perf_graph_str_append_perf(data, strb);
        }

        ucp_proto_perf_graph_str_append_line_break(strb);
        ucs_string_buffer_appendf(strb, "</font>");
    }

    /* Terminate label and node */
    ucs_string_buffer_appendf(strb, ">];\n");

    child_index = 0;
    ucs_array_for_each(child_elem, &perf_node->children) {
        if ((perf_node->type == UCP_PROTO_PERF_NODE_TYPE_SELECT) &&
            (child_index != perf_node->selected_child)) {
            /* This child is not selected */
            child_highlight = -1;
        } else {
            child_highlight = ucs_max(highlight, 0);
        }
        ucp_proto_perf_graph_dump_recurs(*child_elem, id, nodes_hash,
                                         child_highlight, strb);
        ++child_index;
    }
}

static void
ucp_proto_perf_node_graph_dump(const ucp_proto_query_attr_t *proto_attr,
                               const char *file_name,
                               ucp_proto_perf_node_t *node)
{
    khash_t(ucp_proto_graph_node) nodes_hash = KHASH_STATIC_INITIALIZER;
    ucs_string_buffer_t dot_strb             = UCS_STRING_BUFFER_INITIALIZER;
    FILE *fp;

    fp = ucs_open_file("w", UCS_LOG_LEVEL_DIAG, "%s", file_name);
    if (fp == NULL) {
        return;
    }

    ucs_string_buffer_appendf(&dot_strb, "digraph {\n");
    ucs_string_buffer_appendf(
            &dot_strb, "\tnode0 [label=\"%s\\n%s\" shape=box style=rounded]\n",
            proto_attr->desc, proto_attr->config);
    ucp_proto_perf_graph_dump_recurs(node, 0, &nodes_hash, 1, &dot_strb);
    ucs_string_buffer_appendf(&dot_strb, "}\n");

    ucs_string_buffer_dump(&dot_strb, "", fp);
    fclose(fp);

    ucs_string_buffer_cleanup(&dot_strb);
    kh_destroy_inplace(ucp_proto_graph_node, &nodes_hash);
}

static char ucp_proto_debug_fix_filename(char ch)
{
    if ((ch == ']') || (ch == ')') || (ch == '}')) {
        return '\0';
    } else if (isalnum(ch) || (ch == '_')) {
        return ch;
    } else {
        return '_';
    }
}

void
ucp_proto_select_write_info(ucp_worker_h worker,
                            const ucp_proto_select_init_protocols_t *proto_init,
                            const ucs_dynamic_bitmap_t *proto_mask,
                            unsigned selected_idx,
                            ucp_proto_config_t *selected_config,
                            size_t range_start, size_t range_end)
{
    UCS_STRING_BUFFER_ONSTACK(ep_cfg_strb, UCP_PROTO_CONFIG_STR_MAX);
    UCS_STRING_BUFFER_ONSTACK(sel_param_strb, UCP_PROTO_CONFIG_STR_MAX);
    const ucp_proto_init_elem_t *selected_proto, *proto;
    char dir_path[PATH_MAX], file_name[NAME_MAX];
    char range_start_str[64], range_end_str[64];
    const ucp_proto_flat_perf_range_t *range;
    ucp_proto_perf_node_t *select_node;
    ucp_proto_query_attr_t proto_attr;
    size_t selected_child;
    unsigned proto_idx;
    unsigned selected_flags;
    int ret;

    ucp_proto_select_param_dump(worker, selected_config->ep_cfg_index,
                                selected_config->rkey_cfg_index,
                                &selected_config->select_param,
                                ucp_operation_names, &ep_cfg_strb,
                                &sel_param_strb);
    if (!ucp_proto_debug_is_info_enabled(
                worker->context, ucs_string_buffer_cstr(&sel_param_strb))) {
        return;
    }

    ucs_fill_filename_template(worker->context->config.ext.proto_info_dir,
                               dir_path, sizeof(dir_path));
    ret = mkdir(dir_path, S_IRWXU | S_IRGRP | S_IXGRP);
    if ((ret != 0) && (errno != EEXIST)) {
        ucs_debug("failed to create directory %s: %m", dir_path);
        return;
    }

    ucs_string_buffer_translate(&ep_cfg_strb, ucp_proto_debug_fix_filename);
    ucs_string_buffer_translate(&sel_param_strb, ucp_proto_debug_fix_filename);

    selected_proto = &ucs_array_elem(&proto_init->protocols, selected_idx);
    selected_flags = ucp_proto_id_field(selected_proto->proto_id, flags);
    if (selected_flags & UCP_PROTO_FLAG_INVALID) {
        return;
    }

    ucs_memunits_to_str(range_start, range_start_str, sizeof(range_start_str));
    ucs_memunits_to_str(range_end, range_end_str, sizeof(range_end_str));
    selected_child = ucs_dynamic_bitmap_popcount_upto_index(proto_mask,
                                                            selected_idx);
    select_node    = ucp_proto_perf_node_new_select(
            "selected", selected_child, "%s %s..%s",
            ucp_proto_id_field(selected_proto->proto_id, name), range_start_str,
            range_end_str);

    ucs_snprintf_safe(file_name, sizeof(file_name), "%s/%s_%s_%s_%s.dot",
                      dir_path, ucs_string_buffer_cstr(&ep_cfg_strb),
                      ucs_string_buffer_cstr(&sel_param_strb), range_start_str,
                      range_end_str);

    UCS_DYNAMIC_BITMAP_FOR_EACH_BIT(proto_idx, proto_mask) {
        proto = &ucs_array_elem(&proto_init->protocols, proto_idx);
        range = ucp_proto_flat_perf_find_lb(proto->flat_perf, range_start);
        ucs_assert_always(range != NULL);
        ucs_assertv(range->start <= range_start,
                    "range->start=%zu range_start=%zu", range->start,
                    range_start);
        ucs_assertv(range->end >= range_end, "range->end=%zu range_end=%zu",
                    range->end, range_end);

        ucp_proto_perf_node_add_child(select_node, range->node);
    }

    ucp_proto_config_query(worker, selected_config, range_start, &proto_attr);
    ucp_proto_perf_node_graph_dump(&proto_attr, file_name, select_node);

    ucp_proto_perf_node_deref(&select_node);
}

void ucp_proto_select_elem_trace(ucp_worker_h worker,
                                 ucp_worker_cfg_index_t ep_cfg_index,
                                 ucp_worker_cfg_index_t rkey_cfg_index,
                                 const ucp_proto_select_param_t *select_param,
                                 ucp_proto_select_elem_t *select_elem)
{
    ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;
    char *line;

    /* Print human-readable protocol selection table to the log */
    ucp_proto_select_elem_info(worker, ep_cfg_index, rkey_cfg_index,
                               select_param, select_elem, 0, &strb);
    ucs_string_buffer_for_each_token(line, &strb, "\n") {
        ucs_log_print_compact(line);
    }

    ucs_string_buffer_cleanup(&strb);
}
