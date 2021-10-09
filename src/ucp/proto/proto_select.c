/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "proto_select.h"
#include "proto_select.inl"
#include "proto_single.h"

#include <ucp/core/ucp_context.h>
#include <ucp/dt/dt.h>
#include <float.h>

#include <ucp/core/ucp_worker.inl>
#include <ucs/datastruct/array.inl>


/* Compare two protocols which intersect at point X, by examining their value
 * at point (X + UCP_PROTO_MSGLEN_EPSILON)
 */
#define UCP_PROTO_MSGLEN_EPSILON   0.5

/* Threshold for considering two performance values as equal */
#define UCP_PROTO_PERF_EPSILON     1e-15


/* Parameters structure for initializing protocols for a selection parameter */
typedef struct {
    const ucp_proto_select_param_t *select_param; /* Protocol selection parameter */
    ucp_proto_id_mask_t            mask;          /* Which protocols are valid */
    ucp_proto_caps_t               caps[UCP_PROTO_MAX_COUNT]; /* Protocols capabilities */
    void                           *priv_buf;     /* Protocols configuration buffer */
    size_t                         priv_offsets[UCP_PROTO_MAX_COUNT]; /* Offset of each
                                                                         protocol's private
                                                                         area in 'priv_buf' */
} ucp_proto_select_init_protocols_t;

/* Temporary list of constructed protocol thresholds */
typedef struct {
    size_t                         max_length; /* Maximal message size */
    ucp_proto_id_t                 proto_id;   /* Selected protocol up to 'max_length' */
} ucp_proto_threshold_tmp_elem_t;


UCS_ARRAY_DEFINE_INLINE(ucp_proto_thresh, unsigned,
                        ucp_proto_threshold_tmp_elem_t);
UCS_ARRAY_DEFINE_INLINE(ucp_proto_select_range, unsigned,
                        ucp_proto_select_range_t);


const ucp_proto_threshold_elem_t*
ucp_proto_thresholds_search_slow(const ucp_proto_threshold_elem_t *thresholds,
                                 size_t msg_length)
{
    unsigned idx;
    for (idx = 0; msg_length > thresholds[idx].max_msg_length; ++idx);
    return &thresholds[idx];
}

static ucs_status_t
ucp_proto_thresholds_append(ucs_array_t(ucp_proto_thresh) *thresh_list,
                            size_t max_length, ucp_proto_id_t proto_id)
{
    ucp_proto_threshold_tmp_elem_t *thresh_elem;
    ucs_status_t status;

    /* Consolidate with last protocol if possible */
    if (!ucs_array_is_empty(thresh_list)) {
        thresh_elem = ucs_array_last(thresh_list);
        ucs_assert(max_length > thresh_elem->max_length);
        if (thresh_elem->proto_id == proto_id) {
            thresh_elem->max_length = max_length;
            return UCS_OK;
        }
    }

    status = ucs_array_append(ucp_proto_thresh, thresh_list);
    if (status != UCS_OK) {
        return status;
    }

    thresh_elem             = ucs_array_last(thresh_list);
    thresh_elem->max_length = max_length;
    thresh_elem->proto_id   = proto_id;

    return UCS_OK;
}

static int
ucp_proto_select_range_is_equal(const ucp_proto_select_range_t *range_elem1,
                                const ucp_proto_select_range_t *range_elem2)
{
    ucp_proto_perf_type_t perf_type;

    for (perf_type = 0; perf_type < UCP_PROTO_PERF_TYPE_LAST; ++perf_type) {
        if (!ucs_linear_func_is_equal(range_elem1->super.perf[perf_type],
                                      range_elem2->super.perf[perf_type],
                                      UCP_PROTO_PERF_EPSILON)) {
            return 0;
        }
    }

    return range_elem1->cfg_thresh == range_elem2->cfg_thresh;
}

static ucs_status_t
ucp_proto_perf_append(ucs_array_t(ucp_proto_select_range) *range_list,
                      const ucp_proto_select_range_t *new_elem)
{
    ucp_proto_select_range_t *range_elem;
    ucs_status_t status;

    if (!ucs_array_is_empty(range_list)) {
        range_elem = ucs_array_last(range_list);
        ucs_assert(new_elem->super.max_length > range_elem->super.max_length);
        if (ucp_proto_select_range_is_equal(range_elem, new_elem)) {
            range_elem->super.max_length = new_elem->super.max_length;
            return UCS_OK;
        }
    }

    status = ucs_array_append(ucp_proto_select_range, range_list);
    if (status != UCS_OK) {
        return status;
    }

    *ucs_array_last(range_list) = *new_elem;
    return UCS_OK;
}

static void ucp_proto_select_perf_str(const ucs_linear_func_t *perf,
                                      char *time_str, size_t time_str_max,
                                      char *bw_str, size_t bw_str_max)
{
    /* Estimated time per 1024 bytes */
    ucs_snprintf_safe(time_str, time_str_max,
                      UCP_PROTO_PERF_FUNC_TIME_FMT,
                      UCP_PROTO_PERF_FUNC_TIME_ARG(perf));

    /* Estimated bandwidth (MiB/s) */
    ucs_snprintf_safe(bw_str, bw_str_max,
                      UCP_PROTO_PERF_FUNC_BW_FMT,
                      UCP_PROTO_PERF_FUNC_BW_ARG(perf));
}


static ucs_status_t ucp_proto_thresholds_select_best(
        ucp_proto_id_mask_t proto_mask, const ucp_proto_caps_t *proto_caps,
        const ucp_proto_perf_range_t *proto_perf,
        ucp_proto_perf_type_t perf_type,
        ucs_array_t(ucp_proto_thresh) *thresh_list,
        ucs_array_t(ucp_proto_select_range) *range_list, size_t start,
        size_t end)
{
    char time_str[64], bw_str[64], num_str[64];
    ucp_proto_select_range_t range_elem;
    struct {
        ucp_proto_id_t proto_id;
        double         result;
    } curr, best;
    ucs_status_t status;
    double x_intersect;
    size_t midpoint;

    ucs_trace("  %-20s %-20s %-18s", "PROTOCOL", "TIME (ns/KB)", "BANDWIDTH (MB/s)");
    ucs_for_each_bit(curr.proto_id, proto_mask) {
        ucp_proto_select_perf_str(&proto_perf[curr.proto_id].perf[perf_type],
                                  time_str, sizeof(time_str), bw_str,
                                  sizeof(bw_str));
        ucs_trace("  %-20s %-20s %-18s",
                  ucp_proto_id_field(curr.proto_id, name), time_str, bw_str);
    }

    do {
        ucs_assert(proto_mask != 0);

        /* Find best protocol at the 'start' point */
        best.result   = DBL_MAX;
        best.proto_id = UCP_PROTO_ID_INVALID;
        ucs_for_each_bit(curr.proto_id, proto_mask) {
            curr.result = ucs_linear_func_apply(
                    proto_perf[curr.proto_id].perf[perf_type],
                    start + UCP_PROTO_MSGLEN_EPSILON);
            ucs_assert(curr.result != DBL_MAX);
            if ((best.proto_id == UCP_PROTO_ID_INVALID) ||
                (curr.result < best.result)) {
                best = curr;
            }
        }

        /* Since proto_mask != 0, we should find at least one protocol */
        ucs_assert(best.proto_id != UCP_PROTO_ID_INVALID);

        ucs_trace("best protocol at %s is %s",
                  ucs_memunits_to_str(start, num_str, sizeof(num_str)),
                  ucp_proto_id_field(best.proto_id, name));
        ucs_log_indent(1);

        /* Find first (smallest) intersection point between the current best
         * protocol and any other protocol. This would be the point where that
         * other protocol becomes the best one.
         */
        midpoint    = end;
        proto_mask &= ~UCS_BIT(best.proto_id);
        ucs_for_each_bit(curr.proto_id, proto_mask) {
            status = ucs_linear_func_intersect(
                    proto_perf[curr.proto_id].perf[perf_type],
                    proto_perf[best.proto_id].perf[perf_type], &x_intersect);
            if ((status == UCS_OK) && (x_intersect > start)) {
                /* We care only if the intersection is after 'start', since
                 * otherwise best.proto_id is better than curr.proto_id at
                 * 'end' as well as at 'start'.
                 */
                midpoint = ucs_min(ucs_double_to_sizet(x_intersect, SIZE_MAX),
                                   midpoint);
                ucs_memunits_to_str(midpoint, num_str, sizeof(num_str));
                ucs_trace("intersects with %s at %.2f, midpoint is %s",
                          ucp_proto_id_field(curr.proto_id, name), x_intersect,
                          num_str);
            } else {
                ucs_trace("intersects with %s out of range",
                          ucp_proto_id_field(curr.proto_id, name));
            }
        }
        ucs_log_indent(-1);

        status = ucp_proto_thresholds_append(thresh_list, midpoint,
                                             best.proto_id);
        if (status != UCS_OK) {
            return status;
        }

        range_elem.super            = proto_perf[best.proto_id];
        range_elem.super.max_length = midpoint;
        range_elem.cfg_thresh       = proto_caps[best.proto_id].cfg_thresh;
        status = ucp_proto_perf_append(range_list, &range_elem);
        if (status != UCS_OK) {
            return status;
        }

        start = midpoint + 1;
    } while (midpoint < end);

    return UCS_OK;
}

/*
 * Select a protocol for 'msg_length', return last message length for the proto
 */
static ucs_status_t ucp_proto_thresholds_select_next(
        ucp_proto_id_mask_t proto_mask, const ucp_proto_caps_t *proto_caps,
        ucp_proto_perf_type_t perf_type,
        ucs_array_t(ucp_proto_thresh) *thresh_list,
        ucs_array_t(ucp_proto_select_range) *range_list, size_t msg_length,
        size_t *max_length_p)
{
    ucp_proto_id_mask_t valid_proto_mask, disabled_proto_mask;
    ucp_proto_perf_range_t proto_perf[UCP_PROTO_MAX_COUNT];
    ucp_proto_id_t max_prio_proto_id;
    const ucp_proto_caps_t *caps;
    unsigned max_cfg_priority;
    ucp_proto_id_t proto_id;
    ucs_status_t status;
    char range_str[64];
    size_t max_length;
    unsigned i;

    /*
     * Find the valid and configured protocols starting from 'msg_length'.
     * Start with endpoint at SIZE_MAX, and narrow it down whenever we encounter
     * a protocol with different configuration.
     */
    valid_proto_mask    = 0;
    disabled_proto_mask = 0;
    max_cfg_priority    = 0;
    max_length          = SIZE_MAX;
    max_prio_proto_id   = UCP_PROTO_ID_INVALID;
    ucs_for_each_bit(proto_id, proto_mask) {
        caps = &proto_caps[proto_id];

        if (msg_length < caps->min_length) {
            ucs_trace("skipping proto %d with min_length %zu for msg_length %zu",
                      proto_id, caps->min_length, msg_length);
            max_length = ucs_min(max_length, caps->min_length - 1);
            continue;
        }

        /* Update 'max_length' by the maximal message length of the protocol */
        for (i = 0; i < caps->num_ranges; ++i) {
            /* Find first (and only) range which contains 'msg_length' */
            if (msg_length <= caps->ranges[i].max_length) {
                valid_proto_mask    |= UCS_BIT(proto_id);
                proto_perf[proto_id] = caps->ranges[i];
                max_length           = ucs_min(max_length,
                                               caps->ranges[i].max_length);
                break;
            }
        }

        if (!(valid_proto_mask & UCS_BIT(proto_id))) {
            continue;
        }

        /* Apply user threshold configuration */
        if (caps->cfg_thresh != UCS_MEMUNITS_AUTO) {
            if (caps->cfg_thresh == UCS_MEMUNITS_INF) {
                disabled_proto_mask |= UCS_BIT(proto_id);
            } else if (msg_length < caps->cfg_thresh) {
                /* The protocol is lowest priority up to 'cfg_thresh' - 1 */
                disabled_proto_mask |= UCS_BIT(proto_id);
                max_length           = ucs_min(max_length, caps->cfg_thresh - 1);
            } else {
                /* The protocol is force-activated on 'msg_length' and above */
                max_cfg_priority  = ucs_max(max_cfg_priority,
                                            caps->cfg_priority);
                max_prio_proto_id = proto_id;
            }
        }
    }
    ucs_assert(msg_length <= max_length);

    if (valid_proto_mask == 0) {
        return UCS_ERR_UNSUPPORTED;
    }

    ucs_memunits_range_str(msg_length, max_length, range_str,
                           sizeof(range_str));
    ucs_trace("select best protocol for %s", range_str);
    ucs_log_indent(1);

    /* A protocol with configured threshold disables all inferior protocols */
    ucs_for_each_bit(proto_id, valid_proto_mask) {
        if (proto_caps[proto_id].cfg_priority >= max_cfg_priority) {
            continue;
        }

        ucs_assert(max_prio_proto_id != UCP_PROTO_ID_INVALID);
        disabled_proto_mask |= UCS_BIT(proto_id);
        ucs_trace("disable %s with priority %u: prefer %s with priority %u",
                  ucp_proto_id_field(proto_id, name),
                  proto_caps[proto_id].cfg_priority,
                  ucp_proto_id_field(max_prio_proto_id, name),
                  max_cfg_priority);
    }

    /* Remove disabled protocols. 'disabled_proto_mask' must be contained in
     * 'valid_proto_mask'. */
    ucs_assert(!(disabled_proto_mask & ~valid_proto_mask));
    if (valid_proto_mask != disabled_proto_mask) {
        valid_proto_mask &= ~disabled_proto_mask;
    } else {
        /* If all protocols were disabled, we couldn't have any configured
         * protocol (because that protocol would be enabled). In this case we
         * allow using disabled protocols as well.
         */
        ucs_assert(max_cfg_priority == 0);
    }
    ucs_assert(valid_proto_mask != 0);

    status = ucp_proto_thresholds_select_best(valid_proto_mask, proto_caps,
                                              proto_perf, perf_type,
                                              thresh_list, range_list,
                                              msg_length, max_length);
    if (status == UCS_OK) {
        *max_length_p = max_length;
    }

    ucs_log_indent(-1);
    return status;
}

static void
ucp_proto_select_init_trace_caps(ucp_proto_id_t proto_id,
                                 const ucp_proto_init_params_t *init_params)
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
            ucs_trace("range[%d] %s" UCP_PROTO_PERF_FUNC_TYPES_FMT, range_index,
                      ucs_memunits_range_str(range_start, range_end, thresh_str,
                                             sizeof(thresh_str)),
                      UCP_PROTO_PERF_FUNC_TYPES_ARG(perf));
        }
        range_start = range_end + 1;
    }
    ucs_log_indent(-1);
}


static ucs_status_t
ucp_proto_select_init_protocols(ucp_worker_h worker,
                                ucp_worker_cfg_index_t ep_cfg_index,
                                ucp_worker_cfg_index_t rkey_cfg_index,
                                const ucp_proto_select_param_t *select_param,
                                ucp_proto_select_init_protocols_t *proto_init)
{
    ucp_proto_init_params_t init_params;
    ucp_proto_caps_t *proto_caps;
    ucs_string_buffer_t strb;
    size_t priv_size, offset;
    ucp_proto_id_t proto_id;
    ucs_status_t status;
    void *tmp;

    ucs_assert(ep_cfg_index != UCP_WORKER_CFG_INDEX_NULL);

    init_params.worker         = worker;
    init_params.select_param   = select_param;
    init_params.ep_cfg_index   = ep_cfg_index;
    init_params.rkey_cfg_index = rkey_cfg_index;
    init_params.ep_config_key  = &worker->ep_config[ep_cfg_index].key;

    if (rkey_cfg_index == UCP_WORKER_CFG_INDEX_NULL) {
        init_params.rkey_config_key = NULL;
    } else {
        init_params.rkey_config_key = &worker->rkey_config[rkey_cfg_index].key;

        /* rkey configuration must be for the same ep */
        ucs_assertv_always(
                init_params.rkey_config_key->ep_cfg_index == ep_cfg_index,
                "rkey->ep_cfg_index=%d ep_cfg_index=%d",
                init_params.rkey_config_key->ep_cfg_index, ep_cfg_index);
    }

    proto_init->select_param = select_param;
    proto_init->mask         = 0;

    /* Initialize protocols and get their capabilities */
    proto_init->priv_buf = ucs_malloc(ucp_protocols_count * UCP_PROTO_PRIV_MAX,
                                      "ucp_proto_priv");
    if (proto_init->priv_buf == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    offset = 0;
    ucs_for_each_bit(proto_id, worker->context->proto_bitmap) {
        proto_caps             = &proto_init->caps[proto_id];
        init_params.priv       = UCS_PTR_BYTE_OFFSET(proto_init->priv_buf,
                                                     offset);
        init_params.priv_size  = &priv_size;
        init_params.caps       = proto_caps;
        init_params.proto_name = ucp_proto_id_field(proto_id, name);

        ucs_trace("trying %s", ucp_proto_id_field(proto_id, name));
        ucs_log_indent(1);

        status = ucp_proto_id_call(proto_id, init, &init_params);
        if (status != UCS_OK) {
            if (status != UCS_ERR_UNSUPPORTED) {
                ucs_trace("protocol %s failed to initialize: %s",
                          ucp_proto_id_field(proto_id, name),
                          ucs_status_string(status));
            }
            ucs_log_indent(-1);
            continue;
        }

        ucp_proto_select_init_trace_caps(proto_id, &init_params);
        ucs_log_indent(-1);

        /* A successful protocol initialization must return non-empty
         * performance range */
        ucs_assert(proto_caps->min_length < SIZE_MAX);
        ucs_assert(proto_caps->num_ranges > 0);

        proto_init->mask                  |= UCS_BIT(proto_id);
        proto_init->priv_offsets[proto_id] = offset;
        offset                            += priv_size;
    }

    if (proto_init->mask == 0) {
        /* No protocol can support the given selection parameters */
        ucs_string_buffer_init(&strb);
        ucp_proto_select_param_str(select_param, &strb);
        ucs_debug("no protocols found for %s", ucs_string_buffer_cstr(&strb));
        ucs_string_buffer_cleanup(&strb);
        status = UCS_ERR_NO_ELEM;
        goto err_free_priv;
    }

    /* Finalize the shared priv buffer size */
    if (offset == 0) {
        ucs_free(proto_init->priv_buf);
        proto_init->priv_buf = NULL;
    } else {
        tmp = ucs_realloc(proto_init->priv_buf, offset, "ucp_proto_priv");
        if (tmp == NULL) {
            status = UCS_ERR_NO_MEMORY;
            goto err_free_priv;
        }

        proto_init->priv_buf = tmp;
    }

    return UCS_OK;

err_free_priv:
    ucs_free(proto_init->priv_buf);
err:
    return status;
}

static ucs_status_t ucp_proto_select_elem_init_thresh(
        ucp_proto_select_elem_t *select_elem,
        const ucp_proto_select_init_protocols_t *proto_init,
        ucp_proto_perf_type_t perf_type, ucp_worker_cfg_index_t ep_cfg_index,
        ucp_worker_cfg_index_t rkey_cfg_index)
{
    UCS_ARRAY_DEFINE_ONSTACK(tmp_thresh_list, ucp_proto_thresh,
                             UCP_PROTO_MAX_COUNT);
    UCS_ARRAY_DEFINE_ONSTACK(tmp_range_list, ucp_proto_select_range,
                             UCP_PROTO_MAX_PERF_RANGES);
    ucp_proto_select_range_t *perf_ranges, *tmp_range_elem;
    ucp_proto_threshold_tmp_elem_t *tmp_thresh_elem;
    ucp_proto_threshold_elem_t *thresholds;
    ucp_proto_config_t *proto_config;
    size_t msg_length, max_length;
    ucp_proto_id_t proto_id;
    ucs_status_t status;
    size_t priv_offset;
    unsigned i;

    /*
     * Select a protocol for every message size interval, until we cover all
     * possible message sizes until SIZE_MAX.
     */
    msg_length = 0;
    do {
        /* Select a protocol which can handle messages starting from 'msg_length',
         * and update max_length with the last message length for which this
         * protocol is selected.
         */
        status = ucp_proto_thresholds_select_next(proto_init->mask,
                                                  proto_init->caps, perf_type,
                                                  &tmp_thresh_list,
                                                  &tmp_range_list, msg_length,
                                                  &max_length);
        if (status != UCS_OK) {
            if (status == UCS_ERR_UNSUPPORTED) {
                ucs_debug("no protocol for msg_length %zu", msg_length);
            }
            goto err;
        }

        msg_length = max_length + 1;
    } while (max_length < SIZE_MAX);

    /* Set pointer to priv buffer (to release it during cleanup) */
    select_elem->priv_buf   = proto_init->priv_buf;

    ucs_assert_always(!ucs_array_is_empty(&tmp_thresh_list));
    ucs_assert_always(ucs_array_last(&tmp_thresh_list)->max_length ==
                      SIZE_MAX);

    /* Allocate thresholds array */
    thresholds = ucs_calloc(ucs_array_length(&tmp_thresh_list),
                            sizeof(*select_elem->thresholds),
                            "ucp_proto_thresholds");
    if (thresholds == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    select_elem->thresholds = thresholds;

    /* Copy the temporary thresholds list to an array inside select_elem */
    i = 0;
    ucs_array_for_each(tmp_thresh_elem, &tmp_thresh_list) {
        proto_id                     = tmp_thresh_elem->proto_id;
        priv_offset                  = proto_init->priv_offsets[proto_id];
        thresholds[i].max_msg_length = tmp_thresh_elem->max_length;

        proto_config                 = &thresholds[i].proto_config;
        proto_config->select_param   = *proto_init->select_param;
        proto_config->ep_cfg_index   = ep_cfg_index;
        proto_config->rkey_cfg_index = rkey_cfg_index;
        proto_config->proto          = ucp_protocols[proto_id];
        proto_config->priv           = UCS_PTR_BYTE_OFFSET(select_elem->priv_buf,
                                                           priv_offset);
        ++i;
    }

    ucs_assert_always(!ucs_array_is_empty(&tmp_range_list));
    ucs_assert_always(ucs_array_last(&tmp_range_list)->super.max_length ==
                      SIZE_MAX);

    /* Allocate performance functions array */
    perf_ranges = ucs_calloc(ucs_array_length(&tmp_range_list),
                             sizeof(*perf_ranges), "ucp_proto_perf");
    if (perf_ranges == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free_thresholds;
    }

    select_elem->perf_ranges = perf_ranges;

    /* Copy the performance elements */
    i = 0;
    ucs_array_for_each(tmp_range_elem, &tmp_range_list) {
        perf_ranges[i++] = *tmp_range_elem;
    }

    return UCS_OK;

err_free_thresholds:
    ucs_free((void*)select_elem->thresholds);
err:
    return status;
}

static ucp_proto_perf_type_t
ucp_proto_select_param_perf_type(const ucp_proto_select_param_t *select_param)
{
    if (ucp_proto_select_op_attr_from_flags(select_param->op_flags) &
        UCP_OP_ATTR_FLAG_MULTI_SEND) {
        return UCP_PROTO_PERF_TYPE_MULTI;
    } else {
        return UCP_PROTO_PERF_TYPE_SINGLE;
    }
}

static ucs_status_t
ucp_proto_select_elem_init(ucp_worker_h worker,
                           ucp_worker_cfg_index_t ep_cfg_index,
                           ucp_worker_cfg_index_t rkey_cfg_index,
                           const ucp_proto_select_param_t *select_param,
                           ucp_proto_select_elem_t *select_elem)
{
    UCS_STRING_BUFFER_ONSTACK(sel_param_strb, UCP_PROTO_SELECT_PARAM_STR_MAX);
    ucp_proto_select_init_protocols_t *proto_init;
    ucp_proto_perf_type_t perf_type;
    ucs_status_t status;

    ucp_proto_select_param_str(select_param, &sel_param_strb);
    if (rkey_cfg_index != UCP_WORKER_CFG_INDEX_NULL) {
        ucs_string_buffer_appendf(&sel_param_strb, "->");
        ucp_rkey_config_dump_brief(&worker->rkey_config[rkey_cfg_index].key,
                                   &sel_param_strb);
    }
    ucs_trace("worker %p: select protocols ep[%d]/rkey[%d] for %s", worker,
              ep_cfg_index, rkey_cfg_index,
              ucs_string_buffer_cstr(&sel_param_strb));

    ucs_log_indent(1);

    proto_init = ucs_malloc(sizeof(*proto_init), "proto_init");
    if (proto_init == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out;
    }

    status = ucp_proto_select_init_protocols(worker, ep_cfg_index,
                                             rkey_cfg_index, select_param,
                                             proto_init);
    if (status != UCS_OK) {
        goto out_free_proto_init;
    }

    perf_type = ucp_proto_select_param_perf_type(select_param);
    status    = ucp_proto_select_elem_init_thresh(select_elem, proto_init,
                                                  perf_type, ep_cfg_index,
                                                  rkey_cfg_index);
    if (status != UCS_OK) {
        goto err_cleanup_protocols;
    }

    status = UCS_OK;
    goto out_free_proto_init;

err_cleanup_protocols:
    ucs_free(proto_init->priv_buf);
out_free_proto_init:
    ucs_free(proto_init);
out:
    ucs_log_indent(-1);
    return status;
}

static void
ucp_proto_select_elem_cleanup(ucp_proto_select_elem_t *select_elem)
{
    ucs_free((void*)select_elem->perf_ranges);
    ucs_free((void*)select_elem->thresholds);
    ucs_free(select_elem->priv_buf);
}

static void  ucp_proto_select_cache_reset(ucp_proto_select_t *proto_select)
{
    proto_select->cache.key   = UINT64_MAX;
    proto_select->cache.value = NULL;
}

ucp_proto_select_elem_t *
ucp_proto_select_lookup_slow(ucp_worker_h worker,
                             ucp_proto_select_t *proto_select,
                             ucp_worker_cfg_index_t ep_cfg_index,
                             ucp_worker_cfg_index_t rkey_cfg_index,
                             const ucp_proto_select_param_t *select_param)
{
    ucp_proto_select_elem_t *select_elem, tmp_select_elem;
    ucp_proto_select_key_t key;
    ucs_status_t status;
    khiter_t khiter;
    int khret;

    key.param = *select_param;
    khiter    = kh_get(ucp_proto_select_hash, &proto_select->hash, key.u64);
    if (khiter != kh_end(&proto_select->hash)) {
        select_elem = &kh_value(&proto_select->hash, khiter);
        goto out;
    }

    status = ucp_proto_select_elem_init(worker, ep_cfg_index, rkey_cfg_index,
                                        select_param, &tmp_select_elem);
    if (status != UCS_OK) {
        return NULL;
    }

    /* add to hash after initializing the temp element, since calling
     * ucp_proto_select_elem_init() can recursively modify the hash
     */
    khiter = kh_put(ucp_proto_select_hash, &proto_select->hash, key.u64,
                    &khret);
    ucs_assert_always(khret == UCS_KH_PUT_BUCKET_EMPTY);

    select_elem  = &kh_value(&proto_select->hash, khiter);
    *select_elem = tmp_select_elem;

    /* Adding hash values may reallocate the array, so the cached pointer to
     * select_elem may not be valid anymore.
     */
    ucp_proto_select_cache_reset(proto_select);

out:
    return select_elem;
}

ucs_status_t ucp_proto_select_init(ucp_proto_select_t *proto_select)
{
    kh_init_inplace(ucp_proto_select_hash, &proto_select->hash);
    ucp_proto_select_cache_reset(proto_select);
    return UCS_OK;
}

void ucp_proto_select_cleanup(ucp_proto_select_t *proto_select)
{
    ucp_proto_select_elem_t select_elem;

    kh_foreach_value(&proto_select->hash, select_elem,
         ucp_proto_select_elem_cleanup(&select_elem)
    )
    kh_destroy_inplace(ucp_proto_select_hash, &proto_select->hash);
}

static ucs_status_t
ucp_proto_select_dump_all(ucp_worker_h worker,
                          ucp_worker_cfg_index_t ep_cfg_index,
                          ucp_worker_cfg_index_t rkey_cfg_index,
                          const ucp_proto_select_param_t *select_param,
                          ucs_string_buffer_t *strb)
{
    static const char *proto_info_fmt =
            "    %-18s %-18s %-20s %-18s %-12s %s\n";
    ucp_proto_select_init_protocols_t *proto_init;
    ucs_string_buffer_t config_strb;
    ucp_proto_perf_type_t perf_type;
    size_t range_start, range_end;
    const ucp_proto_caps_t *caps;
    ucp_proto_id_t proto_id;
    ucs_status_t status;
    char range_str[64];
    char time_str[64];
    char thresh_str[64];
    char bw_str[64];
    unsigned i;
    void *priv;

    /* Allocate on heap, since the structure is quite large */
    proto_init = ucs_malloc(sizeof(*proto_init), "proto_init");
    if (proto_init == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out;
    }

    status = ucp_proto_select_init_protocols(worker, ep_cfg_index,
                                             rkey_cfg_index, select_param,
                                             proto_init);
    if (status != UCS_OK) {
        goto out_free;
    }

    perf_type = ucp_proto_select_param_perf_type(select_param);

    ucs_string_buffer_appendf(strb, proto_info_fmt, "PROTOCOL", "SIZE",
                              "TIME (nsec)", "BANDWIDTH (MiB/s)", "THRESHOLD",
                              "CONFIGURATION");

    ucs_for_each_bit(proto_id, proto_init->mask) {

        priv = UCS_PTR_BYTE_OFFSET(proto_init->priv_buf,
                                   proto_init->priv_offsets[proto_id]);
        caps = &proto_init->caps[proto_id];

        /* String for configured threshold */
        ucs_memunits_to_str(caps->cfg_thresh, thresh_str, sizeof(thresh_str));

        range_start = caps->min_length;
        for (i = 0; i < caps->num_ranges; ++i) {
            /* String for performance range */
            range_end = caps->ranges[i].max_length;

            ucs_memunits_range_str(range_start, range_end, range_str,
                                   sizeof(range_str));
            ucp_proto_select_perf_str(&caps->ranges[i].perf[perf_type],
                                      time_str, sizeof(time_str), bw_str,
                                      sizeof(bw_str));

            /* Get protocol configuration */
            ucs_string_buffer_init(&config_strb);
            ucp_proto_id_call(proto_id, config_str, range_start, range_end,
                              priv, &config_strb);

            ucs_string_buffer_appendf(
                    strb, proto_info_fmt,
                    (i == 0) ? ucp_proto_id_field(proto_id, name) : "",
                    range_str, time_str, bw_str, (i == 0) ? thresh_str : "",
                    (i == 0) ? ucs_string_buffer_cstr(&config_strb) : "");

            ucs_string_buffer_cleanup(&config_strb);
            range_start = range_end + 1;
        }

    }

    status = UCS_OK;

    ucs_free(proto_init->priv_buf);
out_free:
    ucs_free(proto_init);
out:
    return status;
}

static void
ucp_proto_select_dump_thresholds(const ucp_proto_select_elem_t *select_elem,
                                 ucs_string_buffer_t *strb)
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
    ucs_status_t status;
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

    ucs_string_buffer_appendf(strb, "\n  Candidates:\n");
    status = ucp_proto_select_dump_all(worker, ep_cfg_index, rkey_cfg_index,
                                       select_param, strb);
    if (status != UCS_OK) {
        ucs_string_buffer_appendf(strb, "<Error: %s>\n",
                                  ucs_status_string(status));
    }
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

void ucp_proto_select_short_disable(ucp_proto_select_short_t *proto_short)
{
    proto_short->max_length_unknown_mem = -1;
    proto_short->max_length_host_mem    = -1;
    proto_short->lane                   = UCP_NULL_LANE;
    proto_short->rkey_index             = UCP_NULL_RESOURCE;
}

void
ucp_proto_select_short_init(ucp_worker_h worker, ucp_proto_select_t *proto_select,
                            ucp_worker_cfg_index_t ep_cfg_index,
                            ucp_worker_cfg_index_t rkey_cfg_index,
                            ucp_operation_id_t op_id, uint32_t op_attr_mask,
                            unsigned proto_flags,
                            ucp_proto_select_short_t *proto_short)
{
    ucp_context_h context    = worker->context;
    const ucp_proto_t *proto = NULL;
    const ucp_proto_threshold_elem_t *thresh;
    ucp_proto_select_param_t select_param;
    const ucp_proto_single_priv_t *spriv;
    ucp_memory_info_t mem_info;
    ssize_t max_short_signed;
    uint32_t op_attr;

    ucp_memory_info_set_host(&mem_info);

    /*
     * Find the minimal threshold among all protocols for all possible
     * combinations of bits in 'op_attr_mask'. For example, we are allowed to
     * use fast-path short protocol only if the message size fits short protocol
     * in both regular mode and UCP_OP_ATTR_FLAG_FAST_CMPL mode.
     */
    ucs_log_indent(1);
    ucs_for_each_submask(op_attr, op_attr_mask) {
        ucp_proto_select_param_init(&select_param, op_id, op_attr,
                                    UCP_DATATYPE_CONTIG, &mem_info, 1);
        thresh = ucp_proto_select_lookup(worker, proto_select, ep_cfg_index,
                                         rkey_cfg_index, &select_param, 0);
        if (thresh == NULL) {
            /* no protocol for contig/host */
            goto out_disable;
        }

        ucs_assert(thresh->proto_config.proto != NULL);
        if (!ucs_test_all_flags(thresh->proto_config.proto->flags, proto_flags)) {
            /* the protocol for smallest messages is not short */
            goto out_disable;
        }

        /* If max_msg_length exceeds SSIZE_MAX, use SSIZE_MAX, since short
           protocol thresholds are signed values */
        max_short_signed = ucs_min(thresh->max_msg_length, SSIZE_MAX);

        ucs_trace("found short protocol %s max_msg_length %zu",
                  thresh->proto_config.proto->name, thresh->max_msg_length);

        /* Assume short protocol uses 'ucp_proto_single_priv_t' */
        spriv = thresh->proto_config.priv;

        if (proto == NULL) {
            proto                            = thresh->proto_config.proto;
            proto_short->max_length_host_mem = max_short_signed;
            proto_short->lane                = spriv->super.lane;
            proto_short->rkey_index          = spriv->super.rkey_index;
        } else {
            if ((proto != thresh->proto_config.proto) ||
                (proto_short->lane != spriv->super.lane) ||
                (proto_short->rkey_index != spriv->super.rkey_index)) {
                /* not all op_attr options have same configuration */
                goto out_disable;
            }

            /* Fast-path threshold is the minimal of all op_attr options */
            proto_short->max_length_host_mem =
                    ucs_min(proto_short->max_length_host_mem, max_short_signed);
        }
    }

    /* If we support only host memory, set max short for unknown memory type to
     * be same as for host memory type. Otherwise, disable short if memory type
     * is unknown.
     */
    ucs_assert(proto_short->max_length_host_mem >= 0);
    proto_short->max_length_unknown_mem = (context->num_mem_type_detect_mds > 0) ?
                                          -1 : proto_short->max_length_host_mem;
    ucs_log_indent(-1);
    ucs_trace("%s: short threshold host memory %zd unknown memory %zd",
              ucp_operation_names[op_id], proto_short->max_length_host_mem,
              proto_short->max_length_unknown_mem);
    return;

out_disable:
    ucs_log_indent(-1);
    ucs_trace("%s: disabling short protocol", ucp_operation_names[op_id]);
    ucp_proto_select_short_disable(proto_short);
}

int ucp_proto_select_get_valid_range(
        const ucp_proto_threshold_elem_t *thresholds, size_t *min_length_p,
        size_t *max_length_p)
{
    const ucp_proto_threshold_elem_t *elem;
    size_t max_msg_length;
    int found;

    found         = 0;
    *min_length_p = 0;
    *max_length_p = 0;
    elem          = thresholds;
    do {
        max_msg_length = elem->max_msg_length;
        if (elem->proto_config.proto->flags & UCP_PROTO_FLAG_INVALID) {
            /* Protocol is invalid, so set range start after it */
            if (max_msg_length < SIZE_MAX) {
                *min_length_p = max_msg_length + 1;
            }
        } else {
            /* Protocol is valid, so extend range end */
            *max_length_p = max_msg_length;
            found         = 1;
        }

        ++elem;
    } while (max_msg_length < SIZE_MAX);

    return found;
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

ucp_proto_select_t *
ucp_proto_select_get(ucp_worker_h worker, ucp_worker_cfg_index_t ep_cfg_index,
                     ucp_worker_cfg_index_t rkey_cfg_index,
                     ucp_worker_cfg_index_t *new_rkey_cfg_index)
{
    ucp_rkey_config_key_t rkey_config_key;
    ucs_status_t status;

    if (rkey_cfg_index == UCP_WORKER_CFG_INDEX_NULL) {
        *new_rkey_cfg_index = UCP_WORKER_CFG_INDEX_NULL;
        return &worker->ep_config[ep_cfg_index].proto_select;
    } else {
        rkey_config_key = worker->rkey_config[rkey_cfg_index].key;

        rkey_config_key.ep_cfg_index = ep_cfg_index;
        status = ucp_worker_rkey_config_get(worker, &rkey_config_key, NULL,
                                            new_rkey_cfg_index);
        if (status != UCS_OK) {
            ucs_error("failed to switch to new rkey");
            return NULL;
        }

        return &worker->rkey_config[*new_rkey_cfg_index].proto_select;
    }
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
