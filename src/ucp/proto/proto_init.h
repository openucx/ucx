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


extern const char *ucp_envelope_convex_names[];


/*
 * Accepts a list of performance functions for a given range and appends the
 * convex or concave envelope of these functions to an output list.
 *
 * @param [in] funcs           Array of performance functions.
 * @param [in] funcs_num       Number of functions in list.
 *                             should be considered during envelope calculation.
 * @param [in] range_start     Range interval start.
 * @param [in] range_end       Range interval end.
 * @param [in] convex          Whether to select convex (maximal) or concave
 *                             (minimal) function from 'perf_list'.
 * @param [out] envelope_list  The resulting envelope list. Each entry in this
 *                             array holds the original index in 'perf_list' to
 *                             which it corresponds.
 */
ucs_status_t
ucp_proto_perf_envelope_make(const ucs_linear_func_t *funcs, uint64_t funcs_num,
                             size_t range_start, size_t range_end, int convex,
                             ucp_proto_perf_envelope_t *envelope_list);


ucs_status_t
ucp_proto_init_add_memreg_time(const ucp_proto_common_init_params_t *params,
                               ucp_md_map_t reg_md_map,
                               ucp_proto_perf_factor_id_t cpu_factor_id,
                               const char *perf_node_name, size_t range_start,
                               size_t range_end, ucp_proto_perf_t *perf);


ucs_status_t
ucp_proto_init_add_buffer_copy_time(ucp_worker_h worker, const char *title,
                                    ucs_memory_type_t local_mem_type,
                                    ucs_memory_type_t remote_mem_type,
                                    uct_ep_operation_t memtype_op,
                                    size_t range_start, size_t range_end,
                                    int local, ucp_proto_perf_t *perf);


ucs_status_t ucp_proto_init_perf(const ucp_proto_common_init_params_t *params,
                                 const ucp_proto_common_tl_perf_t *tl_perf,
                                 ucp_proto_perf_node_t *const tl_perf_node,
                                 ucp_md_map_t reg_md_map, const char *perf_name,
                                 ucp_proto_perf_t **perf_p);


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
