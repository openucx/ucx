/*
 * Copyright (C) 2022, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCP_PROTO_DEBUG_H_
#define UCP_PROTO_DEBUG_H_

#include "proto_common.h"
#include "proto_select.h"


/* Format string to display a protocol time */
#define UCP_PROTO_TIME_FMT(_time_var) " " #_time_var ": %.2f ns"
#define UCP_PROTO_TIME_ARG(_time_val) ((_time_val) * 1e9)

/* Format string to display a protocol performance function time */
#define UCP_PROTO_PERF_FUNC_TIME_FMT "%.2f+%.3f*N"
#define UCP_PROTO_PERF_FUNC_TIME_ARG(_perf_func) \
    ((_perf_func)->c * 1e9), ((_perf_func)->m * 1e9 * UCS_KBYTE)

/* Format string to display a protocol performance function bandwidth */
#define UCP_PROTO_PERF_FUNC_BW_FMT "%.2f"
#define UCP_PROTO_PERF_FUNC_BW_ARG(_perf_func) \
    (1.0 / ((_perf_func)->m * UCS_MBYTE))

/* Format string to display a protocol performance function */
#define UCP_PROTO_PERF_FUNC_FMT(_perf_var) " " #_perf_var ": " \
    UCP_PROTO_PERF_FUNC_TIME_FMT " ns/KB, " \
    UCP_PROTO_PERF_FUNC_BW_FMT " MB/s"
#define UCP_PROTO_PERF_FUNC_ARG(_perf_func) \
    UCP_PROTO_PERF_FUNC_TIME_ARG(_perf_func), \
    UCP_PROTO_PERF_FUNC_BW_ARG(_perf_func)

/* Format string to display a protocol performance estimations
 * of different types. See ucp_proto_perf_type_t */
#define UCP_PROTO_PERF_FUNC_TYPES_FMT \
    UCP_PROTO_PERF_FUNC_FMT(single) \
    UCP_PROTO_PERF_FUNC_FMT(multi) \
    UCP_PROTO_PERF_FUNC_FMT(cpu)
#define UCP_PROTO_PERF_FUNC_TYPES_ARG(_perf_func) \
    UCP_PROTO_PERF_FUNC_ARG((&(_perf_func)[UCP_PROTO_PERF_TYPE_SINGLE])), \
    UCP_PROTO_PERF_FUNC_ARG((&(_perf_func)[UCP_PROTO_PERF_TYPE_MULTI])), \
    UCP_PROTO_PERF_FUNC_ARG((&(_perf_func)[UCP_PROTO_PERF_TYPE_CPU]))


/*
 * Protocol performance node types
 */
typedef enum {
    UCP_PROTO_PERF_NODE_TYPE_DATA,   /* Data node */
    UCP_PROTO_PERF_NODE_TYPE_SELECT, /* Select one of children */
    UCP_PROTO_PERF_NODE_TYPE_COMPOSE /* Compose new value from the children */
} ucp_proto_perf_node_type_t;


void ucp_proto_select_perf_str(const ucs_linear_func_t *perf, char *time_str,
                               size_t time_str_max, char *bw_str,
                               size_t bw_str_max);


void ucp_proto_select_init_trace_caps(
        ucp_proto_id_t proto_id, const ucp_proto_init_params_t *init_params);


void ucp_proto_select_info(ucp_worker_h worker,
                           ucp_worker_cfg_index_t ep_cfg_index,
                           ucp_worker_cfg_index_t rkey_cfg_index,
                           const ucp_proto_select_t *proto_select,
                           ucs_string_buffer_t *strb);


void ucp_proto_select_dump_short(const ucp_proto_select_short_t *select_short,
                                 const char *name, ucs_string_buffer_t *strb);


void ucp_proto_select_param_str(const ucp_proto_select_param_t *select_param,
                                const char **operation_names,
                                ucs_string_buffer_t *strb);


void ucp_proto_select_info_str(ucp_worker_h worker,
                               ucp_worker_cfg_index_t rkey_cfg_index,
                               const ucp_proto_select_param_t *select_param,
                               const char **operation_names,
                               ucs_string_buffer_t *strb);


/* Print protocol info to a string buffer */
void ucp_proto_config_info_str(ucp_worker_h worker,
                               const ucp_proto_config_t *proto_config,
                               size_t msg_length, ucs_string_buffer_t *strb);


ucp_proto_perf_node_t *
ucp_proto_perf_node_new_data(const char *name, const char *desc_fmt, ...);


ucp_proto_perf_node_t *
ucp_proto_perf_node_new_select(const char *name, unsigned selected_child,
                               const char *desc_fmt, ...);


ucp_proto_perf_node_t *
ucp_proto_perf_node_new_compose(const char *name, const char *desc_fmt, ...);


void ucp_proto_perf_node_ref(ucp_proto_perf_node_t *perf_node);


void ucp_proto_perf_node_deref(ucp_proto_perf_node_t **perf_node_p);


void ucp_proto_perf_node_own_child(ucp_proto_perf_node_t *perf_node,
                                   ucp_proto_perf_node_t **child_perf_node_p);


void ucp_proto_perf_node_add_child(ucp_proto_perf_node_t *perf_node,
                                   ucp_proto_perf_node_t *child_perf_node);


/* Return the n-th child, or NULL if not exists */
ucp_proto_perf_node_t *
ucp_proto_perf_node_get_child(ucp_proto_perf_node_t *perf_node, unsigned n);


void ucp_proto_perf_node_add_data(ucp_proto_perf_node_t *perf_node,
                                  const char *name,
                                  const ucs_linear_func_t value);


void ucp_proto_perf_node_add_scalar(ucp_proto_perf_node_t *perf_node,
                                    const char *name, double value);


void ucp_proto_perf_node_add_bandwidth(ucp_proto_perf_node_t *perf_node,
                                       const char *name, double value);


const char *ucp_proto_perf_node_name(ucp_proto_perf_node_t *perf_node);


const char *ucp_proto_perf_node_desc(ucp_proto_perf_node_t *perf_node);


/* Replace old_perf_node by new_perf_node and reassign child nodes to it */
void ucp_proto_perf_node_replace(ucp_proto_perf_node_t **old_perf_node_p,
                                 ucp_proto_perf_node_t **new_perf_node_p);


void ucp_proto_select_elem_trace(ucp_worker_h worker,
                                 ucp_worker_cfg_index_t ep_cfg_index,
                                 ucp_worker_cfg_index_t rkey_cfg_index,
                                 const ucp_proto_select_param_t *select_param,
                                 ucp_proto_select_elem_t *select_elem);


#endif
