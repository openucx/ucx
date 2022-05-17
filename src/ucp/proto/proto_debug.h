/*
 * Copyright (C) 2022, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCP_PROTO_DEBUG_H_
#define UCP_PROTO_DEBUG_H_

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
    UCP_PROTO_PERF_FUNC_FMT(multi)
#define UCP_PROTO_PERF_FUNC_TYPES_ARG(_perf_func) \
    UCP_PROTO_PERF_FUNC_ARG((&(_perf_func)[UCP_PROTO_PERF_TYPE_SINGLE])), \
    UCP_PROTO_PERF_FUNC_ARG((&(_perf_func)[UCP_PROTO_PERF_TYPE_MULTI]))


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


void ucp_proto_select_elem_trace(ucp_worker_h worker,
                                 ucp_worker_cfg_index_t ep_cfg_index,
                                 ucp_worker_cfg_index_t rkey_cfg_index,
                                 const ucp_proto_select_param_t *select_param,
                                 ucp_proto_select_elem_t *select_elem);


#endif
