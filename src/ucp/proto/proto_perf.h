/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_PROTO_PERF_H_
#define UCP_PROTO_PERF_H_

#include "proto.h"

#include <ucs/datastruct/linear_func.h>
#include <ucs/datastruct/string_buffer.h>
#include <ucs/type/status.h>


/* Protocol performance data structure over multiple ranges */
typedef struct ucp_proto_perf ucp_proto_perf_t;


/* Protocol performance segment, defines the performance in a single range */
typedef struct ucp_proto_perf_segment ucp_proto_perf_segment_t;


/* Protocol performance factor type */
typedef enum {
    UCP_PROTO_PERF_FACTOR_LOCAL_CPU,
    UCP_PROTO_PERF_FACTOR_REMOTE_CPU,
    UCP_PROTO_PERF_FACTOR_LOCAL_TL,
    UCP_PROTO_PERF_FACTOR_REMOTE_TL,
    UCP_PROTO_PERF_FACTOR_LOCAL_MTYPE_COPY,
    UCP_PROTO_PERF_FACTOR_REMOTE_MTYPE_COPY,
    UCP_PROTO_PERF_FACTOR_LAST_WO_LATENCY,
    UCP_PROTO_PERF_FACTOR_LATENCY = UCP_PROTO_PERF_FACTOR_LAST_WO_LATENCY,
    UCP_PROTO_PERF_FACTOR_LAST
} ucp_proto_perf_factor_id_t;


/* Final protocol performance segment, represents estimated protocol performance
 * in a single range */
typedef struct {
    size_t                start;
    size_t                end;
    ucs_linear_func_t     value;
    ucp_proto_perf_node_t *node;
} ucp_proto_flat_perf_range_t;

/* Structure that stores final protocols performance for comparison */
UCS_ARRAY_DECLARE_TYPE(ucp_proto_flat_perf_t, unsigned,
                       ucp_proto_flat_perf_range_t);


/* Array of all performance factors */
typedef ucs_linear_func_t ucp_proto_perf_factors_t[UCP_PROTO_PERF_FACTOR_LAST];


#define UCP_PROTO_PERF_FACTORS_INITIALIZER {}


/* Iterate on all segments within a given range */
#define ucp_proto_perf_segment_foreach_range(_seg, _seg_start, _seg_end, \
                                             _perf, _range_start, _range_end) \
    for (_seg = ucp_proto_perf_find_segment_lb(_perf, _range_start); \
         (_seg != NULL) && \
         (_seg_start = \
                  ucs_max(_range_start, ucp_proto_perf_segment_start(seg)), \
         _seg_end = ucs_min(_range_end, ucp_proto_perf_segment_end(seg)), \
         _seg_start <= seg_end); \
         _seg = ucp_proto_perf_segment_next(_perf, _seg))


/**
 * Initialize a new performance data structure.
 *
 * @param [in]  name        Name of the performance data structure.
 * @param [out] perf_p      Filled with the new performance data structure.
 */
ucs_status_t ucp_proto_perf_create(const char *name, ucp_proto_perf_t **perf_p);


/**
 * Destroy a performance data structure and free associated memory.
 * The reference counts of any perf_node objects passed to
 * @ref ucp_proto_perf_add_func() will be adjusted accordingly.
 *
 * @param [in]  perf        Performance data structure to destroy.
*/
void ucp_proto_perf_destroy(ucp_proto_perf_t *perf);


/**
 * @return Whether the perf is empty.
 */
int ucp_proto_perf_is_empty(const ucp_proto_perf_t *perf);


/**
 * @return Perf structure name.
 */
const char *ucp_proto_perf_name(const ucp_proto_perf_t *perf);


/**
 * Add linear functions to several performance factors at the range
 * [ @a start, @a end ]. The performance functions to add are provided in the
 * array @a funcs, each entry corresponding to a factor id defined in
 * @ref ucp_proto_perf_factor_id_t.
 *
 * Initially, all ranges are uninitialized; repeated calls to this function
 * should be used to populate the @a perf data structure.
 *
 * @param [in] perf            Performance data structure to update.
 * @param [in] start           Add the performance function to this range start (inclusive).
 * @param [in] end             Add the performance function to this range end (inclusive).
 * @param [in] perf_factors    Array of performance functions to add.
 * @param [in] child_perf_node Performance node that would be considered as
 *                             child node for all segment nodes on [start, end]
 *                             interval.
 * @param [in] title           Title for performance node that would be created
 *                             to represent @a perf_factors data.
 * @param [in] desc_fmt        Formatted description for performance node that
 *                             would be created to represent @a perf_factors data.
 *
 * @note This function may adjust the reference count of @a perf_node as needed.
 */
ucs_status_t
ucp_proto_perf_add_funcs(ucp_proto_perf_t *perf, size_t start, size_t end,
                         const ucp_proto_perf_factors_t perf_factors,
                         ucp_proto_perf_node_t *child_perf_node,
                         const char *title, const char *desc_fmt, ...);


/**
 * Create a proto perf structure that is the aggregation of multiple other perf
 * structures: In the ranges where ALL given perf structures are defined, the
 * result is the factor-wise sum of the performance values. The performance node
 * of the resulting range will be the parent of the respective ranges in the
 * provided perf structures.
 * Other ranges, where at least one of the given perf structures is not defined,
 * will also not be defined in the result.
 *
 * @param [in]  name        Name of the performance data structure.
 * @param [in]  perf_elems  Array of pointers to the performance structures
 *                          that should be aggregated.
 * @param [in]  num_elems   Number of elements in @a perf_elems array.
 * @param [out] perf_p      Filled with the new performance data structure.
 */
ucs_status_t ucp_proto_perf_aggregate(const char *name,
                                      const ucp_proto_perf_t *const *perf_elems,
                                      unsigned num_elems,
                                      ucp_proto_perf_t **perf_p);

/**
 * @ref ucp_proto_perf_aggregate() for two perf structures
 */
ucs_status_t ucp_proto_perf_aggregate2(const char *name,
                                       const ucp_proto_perf_t *perf1,
                                       const ucp_proto_perf_t *perf2,
                                       ucp_proto_perf_t **perf_p);


/**
 * Expand given perf by estimation that all messages on interval
 * [end of @a frag_seg + 1, @a max_length] would be sent in a pipeline async
 * manner using data provided by @a frag_seg as a performance for sending one
 * fragment.
 *
 * To understand what does it mean, please see the following example:
 *
 * 3-factor 3-msg pipeline:
 * 1 msg: [=1=] [======2======] [=3=]
 * 2 msg:       [=1=]           [======2======] [=3=]
 * 3 msg:             [=1=]                     [======2======] [=3=]
 * Approximation:
 *        [=1=] [======================2======================] [=3=]
 * 
 * All the factors except longest one turn into constant fragment overhead
 * due to overlapping (1 and 3 from example).
 *
 * Longest factor still saves the linear function part but has additional
 * overhead turned to dynamic since it starts to depend on number of sent
 * fragments.
 * (2 from example).
 * 
 * LATENCY factor cannot be chosen as longest one since it overlaps with
 * other simultaneous LATENCY factor operations.
 *
 * @param [in] perf       Performance data structure which includes fragment
 *                        performance.
 * @param [in] ppln_perf  Performance data structure which will be extended
 *                        by pipeline performance.
 * @param [in] max_length Message size until what @a perf would be updated.
 * 
 * @return NULL in case of error, last segment of `perf` which was used as
 *         performance estimation for sending one fragment.
 */
const ucp_proto_perf_segment_t *
ucp_proto_perf_add_ppln(const ucp_proto_perf_t *perf,
                        ucp_proto_perf_t *ppln_perf, size_t max_length);


/**
 * Create a proto perf structure based on @a remote_perf, converting the values
 * of local factors to remote ones and vice versa.
 *
 * @param [in]  remote_perf Performance data structure to turn.
 * @param [out] perf_p      Filled with the new performance data structure.
 */
ucs_status_t ucp_proto_perf_remote(const ucp_proto_perf_t *remote_perf,
                                   ucp_proto_perf_t **perf_p);


/**
 * Convert given @a perf to @a flat_perf structure that contains convex or
 * concave envelope across all factors for each segment. Used for async scheme
 * performance estimations.
 *
 * @param [in]  perf          Performance data structure to convert.
 * @param [in]  convex        If 1 calculate convex, if 0 concave.
 * @param [out] flat_perf_ptr Filled with convex or concave envelope.
 */
ucs_status_t ucp_proto_perf_envelope(const ucp_proto_perf_t *perf, int convex,
                                     ucp_proto_flat_perf_t **flat_perf_ptr);


/**
 * Convert given @a perf to @a flat_perf structure that contains sum of all
 * factors for each segment. Used for blocking scheme performance estimations.
 *
 * @param [in]  perf          Performance data structure to convert.
 * @param [out] flat_perf_ptr Filled with sum of all factors.
 */
ucs_status_t ucp_proto_perf_sum(const ucp_proto_perf_t *perf,
                                ucp_proto_flat_perf_t **flat_perf_ptr);


/**
 * Find the first segment that contains a point greater than or equal to a given
 * lower bound value.
 *
 * @param [in] perf          Performance data structure.
 * @param [in] lb            Lower bound of the segment to find.
 *
 * @return Pointer to the first segment that contains a point greater than or
 *         equal to @a lb, or NULL if all segments end before @a lb.
 */
ucp_proto_perf_segment_t *
ucp_proto_perf_find_segment_lb(const ucp_proto_perf_t *perf, size_t lb);


/**
 * Get the performance function of a given factor at a given segment.
 *
 * @param [in] seg           Segment to get the performance function from.
 * @param [in] factor_id     Performance factor id.
 *
 * @return The performance function of @a factor_id at @a seg.
 */
ucs_linear_func_t
ucp_proto_perf_segment_func(const ucp_proto_perf_segment_t *seg,
                            ucp_proto_perf_factor_id_t factor_id);


/**
 * Get the start point of a given segment.
 *
 * @param [in] seg           Segment to get the start value from.
 *
 * @return The start point of @a seg.
 */
size_t ucp_proto_perf_segment_start(const ucp_proto_perf_segment_t *seg);


/**
 * Get end point of a given segment.
 *
 * @param [in] seg           Segment to get the end value from.
 *
 * @return The end point of @a seg.
 */
size_t ucp_proto_perf_segment_end(const ucp_proto_perf_segment_t *seg);


/**
 * Get the performance node of a given segment.
 *
 * @param [in] seg           Segment to get the performance node from.
 *
 * @return The performance node of @a seg.
 */
ucp_proto_perf_node_t *
ucp_proto_perf_segment_node(const ucp_proto_perf_segment_t *seg);


/**
 * Get next segment, or NULL if none.
 */
const ucp_proto_perf_segment_t *
ucp_proto_perf_segment_next(const ucp_proto_perf_t *perf,
                            const ucp_proto_perf_segment_t *seg);


/**
 * Dump the performance data of the segment to a string buffer.
 *
 * @param [in]  seg         Segment to dump.
 * @param [out] strb        String buffer to dump the performance data to.
 */
void ucp_proto_perf_segment_str(const ucp_proto_perf_segment_t *seg,
                                ucs_string_buffer_t *strb);


/**
 * Dump the final performance data structure to a string buffer.
 *
 * @param [in]  flat_perf   Final performance data structure to dump.
 * @param [out] strb        String buffer to dump the performance data to.
 */
void ucp_proto_flat_perf_str(const ucp_proto_flat_perf_t *flat_perf,
                             ucs_string_buffer_t *strb);


/**
 * Dump the performance data structure to a string buffer.
 *
 * @param [in]  perf        Performance data structure to dump.
 * @param [out] strb        String buffer to dump the performance data to.
 */
void ucp_proto_perf_str(const ucp_proto_perf_t *perf,
                        ucs_string_buffer_t *strb);


/**
 * Find the first range that contains a point greater than or equal to a given
 * lower bound value.
 *
 * @param [in] flat_perf     Flat performance data structure.
 * @param [in] lb            Lower bound of the segment to find.
 *
 * @return Pointer to the first range that contains a point greater than or
 *         equal to @a lb, or NULL if all ranges end before @a lb.
 */
const ucp_proto_flat_perf_range_t *
ucp_proto_flat_perf_find_lb(const ucp_proto_flat_perf_t *flat_perf, size_t lb);


/**
 * Destroy a flat performance data structure and free associated memory.
 * The reference counts of any perf_node objects passed to
 * @ref ucp_proto_perf_add_func() will be adjusted accordingly.
 *
 * @param [in]  flat_perf   Performance data structure to destroy.
*/
void ucp_proto_flat_perf_destroy(ucp_proto_flat_perf_t *flat_perf);


/**
 * Get factor for CPU operations.
 */
static UCS_F_ALWAYS_INLINE ucp_proto_perf_factor_id_t
ucp_proto_buffer_copy_cpu_factor_id(int is_local)
{
    return is_local ? UCP_PROTO_PERF_FACTOR_LOCAL_CPU :
                      UCP_PROTO_PERF_FACTOR_REMOTE_CPU;
}

#endif
