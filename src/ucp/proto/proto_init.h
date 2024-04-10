/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2021. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_PROTO_INIT_H_
#define UCP_PROTO_INIT_H_

#include "proto.h"
#include "proto_select.h"
#include "proto_common.h"

#include <ucs/datastruct/array.h>


/* Element of list of trends to select next best one */
typedef struct {
    size_t   max_length;  /* Maximal message size */
    unsigned index;       /* Selected index of the input array */
} ucp_proto_perf_envelope_elem_t;


UCS_ARRAY_DECLARE_TYPE(ucp_proto_perf_envelope_t, unsigned,
                       ucp_proto_perf_envelope_elem_t);
UCS_ARRAY_DECLARE_TYPE(ucp_proto_perf_list_t, unsigned, ucs_linear_func_t);
UCS_ARRAY_DECLARE_TYPE(ucp_proto_perf_ranges_t, unsigned,
                       ucp_proto_perf_range_t);

/**
 * Add a "pipelined performance" range, which represents the send time of
 * multiples fragments. 'frag_range' is the time to send a single fragment.
 */
void ucp_proto_common_add_ppln_range(ucp_proto_caps_t *caps,
                                     const ucp_proto_perf_range_t *frag_range,
                                     size_t max_length);


void ucp_proto_common_init_base_caps(
        const ucp_proto_common_init_params_t *params, ucp_proto_caps_t *caps,
        size_t min_length);


void ucp_proto_perf_range_add_data(const ucp_proto_perf_range_t *range);


/*
 * Accepts a list of performance functions for a given range and appends the
 * convex or concave envelope of these functions to an output list.
 *
 * @param [in] perf_list       List of performance functions.
 * @param [in] range_start     Range interval start.
 * @param [in] range_end       Range interval end.
 * @param [in] convex          Whether to select convex (maximal) or concave
 *                             (minimal) function from 'perf_list'.
 * @param [out] envelope_list  The resulting envelope list. Each entry in this
 *                             array holds the original index in 'perf_list' to
 *                             which it corresponds.
 */
ucs_status_t
ucp_proto_perf_envelope_make(const ucp_proto_perf_list_t *perf_list,
                             size_t range_start, size_t range_end, int convex,
                             ucp_proto_perf_envelope_t *envelope_list);


/**
 * Initialize the performance of a protocol that consists of several parallel
 * stages. The performance estimations are added to params->caps.
 *
 * @param [in]    proto_name    Protocol name, for debugging.
 * @param [in]    range_start   Range interval start.
 * @param [in]    range_end     Range interval end.
 * @param [in]    bias          Performance bias (0 - no bias).
 * @param [in]    stages        Array of parallel stages performance ranges.
 * @param [in]    num_stages    Number of parallel stages in the protocol.
 * @param [inout] caps          Filled with protocol performance data.
 */
ucs_status_t
ucp_proto_init_parallel_stages(const char *proto_name, size_t range_start,
                               size_t range_end, double bias,
                               const ucp_proto_perf_range_t **stages,
                               unsigned num_stages, ucp_proto_caps_t *caps);


void ucp_proto_init_memreg_time(const ucp_proto_common_init_params_t *params,
                                ucp_md_map_t reg_md_map,
                                ucs_linear_func_t *memreg_time,
                                ucp_proto_perf_node_t **perf_node_p);

ucs_status_t
ucp_proto_init_buffer_copy_time(ucp_worker_h worker, const char *title,
                                ucs_memory_type_t local_mem_type,
                                ucs_memory_type_t remote_mem_type,
                                uct_ep_operation_t memtype_op,
                                ucs_linear_func_t *copy_time,
                                ucp_proto_perf_node_t **perf_node_p);


ucs_status_t
ucp_proto_common_init_caps(const ucp_proto_common_init_params_t *params,
                           const ucp_proto_common_tl_perf_t *tl_perf,
                           ucp_proto_perf_node_t *const tl_perf_node,
                           ucp_md_map_t reg_md_map, ucp_proto_caps_t *caps);


/**
 * Check if protocol initialization parameters contain one of the specified
 * operations.
 *
 * @param [in] init_params   Protocol initialization parameters.
 * @param [in] op_id_mask    Bitmap of operations to check.
 *
 * @return Nonzero if one of the operations is present.
 */
int ucp_proto_init_check_op(const ucp_proto_init_params_t *init_params,
                            uint64_t op_id_mask);

#endif
