/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
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
    uint64_t                 id;
    ucp_proto_select_range_t range;
} ucp_proto_perf_envelope_elem_t;


UCS_ARRAY_DECLARE_TYPE(ucp_proto_perf_envelope, unsigned,
                       ucp_proto_perf_envelope_elem_t);
UCS_ARRAY_DECLARE_FUNCS(ucp_proto_perf_envelope, unsigned,
                        ucp_proto_perf_envelope_elem_t,);


typedef ucs_array_t(ucp_proto_perf_envelope) ucp_proto_perf_envelope_t;


/**
 * Add a "pipelined performance" range, which represents the send time of
 * multiples fragments. 'frag_range' is the time to send a single fragment.
 */
void ucp_proto_common_add_ppln_range(const ucp_proto_init_params_t *init_params,
                                     const ucp_proto_perf_range_t *frag_range,
                                     size_t max_length);


void ucp_proto_common_init_base_caps(
        const ucp_proto_common_init_params_t *params, size_t min_length);


/*
 * Accepts list of operations that happen in parallel and create a ranges
 * on min..max which represents their overall performance.
 */
ucs_status_t ucp_proto_common_add_perf_ranges(
        const ucp_proto_common_init_params_t *params,
        size_t min_length, size_t max_length,
        const ucp_proto_perf_envelope_t *parallel_statge_list);


void ucp_proto_perf_envelope_append(ucp_proto_perf_envelope_t *list,
                                    const char *name,
                                    const ucp_proto_perf_range_t *range,
                                    size_t frag_size, ucs_linear_func_t bias);


/*
 * Accepts a list of performance functions for a given range
 * (min_length..max_length), and appends the convex (or concave)
 * envelope segments of these functions to the 'envelope_list'. The envelope
 * is the minimal (or maximal) function at each point in that range.
 */
ucs_status_t
ucp_proto_perf_envelope_make(const ucp_proto_perf_envelope_t *list,
                             ucp_proto_perf_envelope_t *envelope_list,
                             size_t min_length, size_t max_length,
                             ucp_proto_perf_type_t perf_type, int convex);


ucs_status_t
ucp_proto_common_init_caps(const ucp_proto_common_init_params_t *params,
                           const ucp_proto_common_tl_perf_t *perf,
                           ucp_md_map_t reg_md_map);

#endif
