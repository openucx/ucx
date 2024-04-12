/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "proto_init.h"
#include "proto_debug.h"
#include "proto_single.h"
#include "proto_select.inl"

#include <ucp/core/ucp_context.h>
#include <ucp/dt/dt.h>
#include <ucs/datastruct/array.h>
#include <ucs/datastruct/dynamic_bitmap.h>

#include <ucp/core/ucp_worker.inl>


typedef struct {
    ucp_proto_id_t   proto_id;
    size_t           priv_offset;
    size_t           cfg_thresh; /* Configured protocol threshold */
    unsigned         cfg_priority; /* Priority of configuration */
    ucp_proto_caps_t caps;
} ucp_proto_init_elem_t;

/* Parameters structure for initializing protocols for a selection parameter */
struct ucp_proto_probe_ctx {
    ucs_array_s(size_t, uint8_t)                 priv_buf;
    ucs_array_s(unsigned, ucp_proto_init_elem_t) protocols;
};

typedef ucp_proto_probe_ctx_t ucp_proto_select_init_protocols_t;

UCS_ARRAY_DECLARE_TYPE(ucp_proto_ranges_t, unsigned, ucp_proto_perf_range_t);
UCS_ARRAY_DECLARE_TYPE(ucp_proto_thresh_t, unsigned,
                       ucp_proto_threshold_elem_t);

const ucp_proto_threshold_elem_t*
ucp_proto_thresholds_search_slow(const ucp_proto_threshold_elem_t *thresholds,
                                 size_t msg_length)
{
    unsigned idx;
    for (idx = 0; msg_length > thresholds[idx].max_msg_length; ++idx);
    return &thresholds[idx];
}

static const ucp_proto_perf_range_t *
ucp_proto_caps_range_find(const ucp_proto_caps_t *caps, size_t msg_length)
{
    const ucp_proto_perf_range_t *range;

    ucs_carray_for_each(range, caps->ranges, caps->num_ranges) {
        if (msg_length <= range->max_length) {
            return range;
        }
    }

    return NULL;
}

static const void *ucp_proto_select_init_priv_buf(
        const ucp_proto_select_init_protocols_t *proto_init, unsigned proto_idx)
{
    size_t priv_offset =
            ucs_array_elem(&proto_init->protocols, proto_idx).priv_offset;
    return &ucs_array_elem(&proto_init->priv_buf, priv_offset);
}

/*
 * Fills 'proto_mask' and 'perf_list' with candidate protocols for the next
 * range, and sets *max_length_p to the end of that range.
 */
static ucs_status_t ucp_proto_thresholds_next_range(
        const ucp_proto_select_init_protocols_t *proto_init,
        const ucp_proto_select_param_t *select_param, size_t msg_length,
        ucp_proto_perf_list_t *perf_list, size_t *max_length_p,
        ucs_dynamic_bitmap_t *proto_mask)
{
    char range_str[64], time_str[64], bw_str[64];
    ucs_dynamic_bitmap_t disabled_proto_mask;
    const ucp_proto_perf_range_t *range;
    const ucp_proto_init_elem_t *proto;
    const char *max_prio_proto_name;
    ucp_proto_perf_type_t perf_type;
    unsigned max_cfg_priority;
    ucs_status_t status;
    unsigned proto_idx;
    size_t max_length;

    /*
     * Find the valid and configured protocols starting from 'msg_length'.
     * Start with endpoint at SIZE_MAX, and narrow it down whenever we encounter
     * a protocol with different configuration.
     */
    max_cfg_priority    = 0;
    max_length          = SIZE_MAX;
    max_prio_proto_name = NULL;
    ucs_dynamic_bitmap_reset_all(proto_mask);
    ucs_dynamic_bitmap_init(&disabled_proto_mask);

    for (proto_idx = 0; proto_idx < ucs_array_length(&proto_init->protocols);
         ++proto_idx) {
        proto = &ucs_array_elem(&proto_init->protocols, proto_idx);
        if (msg_length < proto->caps.min_length) {
            ucs_trace(
                    "skipping proto %s with min_length %zu for msg_length %zu",
                    ucp_proto_id_field(proto->proto_id, name),
                    proto->caps.min_length, msg_length);
            max_length = ucs_min(max_length, proto->caps.min_length - 1);
            continue;
        }

        /* Update 'max_length' by the maximal message length of the protocol */
        range = ucp_proto_caps_range_find(&proto->caps, msg_length);
        if (range == NULL) {
            continue;
        }

        max_length = ucs_min(max_length, range->max_length);
        ucs_dynamic_bitmap_set(proto_mask, proto_idx);

        /* Apply user threshold configuration */
        if (proto->cfg_thresh != UCS_MEMUNITS_AUTO) {
            if (proto->cfg_thresh == UCS_MEMUNITS_INF) {
                ucs_dynamic_bitmap_set(&disabled_proto_mask, proto_idx);
            } else if (msg_length < proto->cfg_thresh) {
                /* The protocol is lowest priority up to 'cfg_thresh' - 1 */
                ucs_dynamic_bitmap_set(&disabled_proto_mask, proto_idx);
                max_length = ucs_min(max_length, proto->cfg_thresh - 1);
            } else if (proto->cfg_priority >= max_cfg_priority) {
                /* The protocol is force-activated on 'msg_length' and above */
                max_cfg_priority    = proto->cfg_priority;
                max_prio_proto_name = ucp_proto_id_field(proto->proto_id, name);
            }
        }
    }
    ucs_assert(msg_length <= max_length);

    if (ucs_dynamic_bitmap_is_zero(proto_mask)) {
        status = UCS_ERR_UNSUPPORTED;
        goto out;
    }

    ucs_memunits_range_str(msg_length, max_length, range_str,
                           sizeof(range_str));
    ucs_trace("select best protocol for %s %s",
              ucp_operation_names[ucp_proto_select_op_id(select_param)],
              range_str);

    ucs_log_indent(1);

    /* A protocol with configured threshold disables all inferior protocols */
    UCS_DYNAMIC_BITMAP_FOR_EACH_BIT(proto_idx, proto_mask) {
        proto = &ucs_array_elem(&proto_init->protocols, proto_idx);
        if (proto->cfg_priority >= max_cfg_priority) {
            continue;
        }

        ucs_assert_always(max_prio_proto_name != NULL);
        ucs_dynamic_bitmap_set(&disabled_proto_mask, proto_idx);
        /* coverity[overrun-local] */
        ucs_trace("disable %s with priority %u: prefer %s with priority %u",
                  ucp_proto_id_field(proto->proto_id, name),
                  proto->cfg_priority, max_prio_proto_name, max_cfg_priority);
    }

    /* Remove disabled protocols. 'disabled_proto_mask' must be contained in
     * 'valid_proto_mask'. */
    if (ucs_dynamic_bitmap_is_equal(proto_mask, &disabled_proto_mask)) {
        /* If all protocols were disabled, we couldn't have any configured
         * protocol (because that protocol would be enabled). In this case we
         * allow using disabled protocols as well.
         */
        ucs_assert(max_cfg_priority == 0);
    } else {
        ucs_dynamic_bitmap_not_inplace(&disabled_proto_mask,
                                       ucs_dynamic_bitmap_num_bits(proto_mask));
        ucs_dynamic_bitmap_and_inplace(proto_mask, &disabled_proto_mask);
    }
    ucs_assert(!ucs_dynamic_bitmap_is_zero(proto_mask));

    /* Add data to the performance tree */
    perf_type = ucp_proto_select_param_perf_type(select_param);
    UCS_DYNAMIC_BITMAP_FOR_EACH_BIT(proto_idx, proto_mask) {
        proto = &ucs_array_elem(&proto_init->protocols, proto_idx);
        range = ucp_proto_caps_range_find(&proto->caps, msg_length);

        ucp_proto_select_perf_str(&range->perf[perf_type], time_str,
                                  sizeof(time_str), bw_str, sizeof(bw_str));
        ucs_trace("  %-20s %-20s %-18s",
                  ucp_proto_id_field(proto->proto_id, name), time_str, bw_str);

        *ucs_array_append(perf_list, status = UCS_ERR_NO_MEMORY;
                          goto out_unindent) = range->perf[perf_type];
    }

    status        = UCS_OK;
    *max_length_p = max_length;

out_unindent:
    ucs_log_indent(-1);
out:
    ucs_dynamic_bitmap_cleanup(&disabled_proto_mask);
    return status;
}

static ucs_status_t
ucp_proto_select_init_protocols(ucp_worker_h worker,
                                ucp_worker_cfg_index_t ep_cfg_index,
                                ucp_worker_cfg_index_t rkey_cfg_index,
                                const ucp_proto_select_param_t *select_param,
                                ucp_proto_select_init_protocols_t *proto_init)
{
    ucp_proto_caps_t proto_caps = {};
    UCS_STRING_BUFFER_ONSTACK(strb, UCP_PROTO_CONFIG_STR_MAX);
    void *proto_priv = ucs_alloca(UCP_PROTO_PRIV_MAX);
    ucp_proto_init_params_t init_params;
    ucs_status_t status;
    size_t priv_size;

    ucs_assert(ep_cfg_index != UCP_WORKER_CFG_INDEX_NULL);

    init_params.worker         = worker;
    init_params.select_param   = select_param;
    init_params.ep_cfg_index   = ep_cfg_index;
    init_params.rkey_cfg_index = rkey_cfg_index;
    init_params.ep_config_key  = &ucs_array_elem(&worker->ep_config,
                                                 ep_cfg_index).key;
    init_params.ctx            = proto_init;

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

    ucs_array_init_dynamic(&proto_init->protocols);
    ucs_array_init_dynamic(&proto_init->priv_buf);

    ucs_for_each_bit(init_params.proto_id, worker->context->proto_bitmap) {
        ucs_assert(init_params.proto_id < ucp_protocols_count()); /* coverity */
        init_params.priv       = proto_priv;
        init_params.priv_size  = &priv_size;
        init_params.caps       = &proto_caps;
        init_params.proto_name = ucp_proto_id_field(init_params.proto_id, name);

        ucs_log_indent(1);
        if (ucp_proto_id_field(init_params.proto_id, probe) != NULL) {
            ucs_trace("probing %s", init_params.proto_name);
            ucp_proto_id_call(init_params.proto_id, probe, &init_params);
        } else if (ucp_proto_id_field(init_params.proto_id, init) != NULL) {
            ucs_trace("initializing %s", init_params.proto_name);
            status = ucp_proto_id_call(init_params.proto_id, init,
                                       &init_params);
            if (status == UCS_OK) {
                ucp_proto_select_add_proto(&init_params, proto_caps.cfg_thresh,
                                           proto_caps.cfg_priority, &proto_caps,
                                           proto_priv, priv_size);
            }
        } else {
            ucs_fatal("protocol '%s' both init and probe are NULL",
                      init_params.proto_name);
        }
        ucs_log_indent(-1);
    }

    if (ucs_array_is_empty(&proto_init->protocols)) {
        /* No protocol can support the given selection parameters */
        ucp_proto_select_param_str(select_param, ucp_operation_names, &strb);
        ucs_debug("no protocols found for %s", ucs_string_buffer_cstr(&strb));
        ucs_array_cleanup_dynamic(&proto_init->priv_buf);
        ucs_array_cleanup_dynamic(&proto_init->protocols);
        return UCS_ERR_NO_ELEM;
    }

    return UCS_OK;
}

static void
ucp_proto_select_perf_ranges_cleanup(ucp_proto_perf_range_t *perf_ranges,
                                     unsigned num_ranges)
{
    ucp_proto_perf_range_t *range;

    ucs_assert(perf_ranges != NULL); /* For coverity */
    ucs_carray_for_each(range, perf_ranges, num_ranges) {
        ucp_proto_perf_node_deref(&range->node);
    }
}

void ucp_proto_select_caps_reset(ucp_proto_caps_t *caps)
{
    caps->cfg_thresh   = UCS_MEMUNITS_AUTO;
    caps->cfg_priority = 0;
    caps->min_length   = 0;
    caps->num_ranges   = 0;
}

void ucp_proto_select_caps_cleanup(ucp_proto_caps_t *caps)
{
    ucp_proto_select_perf_ranges_cleanup(caps->ranges, caps->num_ranges);
    ucp_proto_select_caps_reset(caps);
}

static void
ucp_proto_select_cleanup_protocols(ucp_proto_select_init_protocols_t *proto_init)
{
    ucp_proto_init_elem_t *init_elem;

    ucs_array_for_each(init_elem, &proto_init->protocols) {
        ucp_proto_select_caps_cleanup(&init_elem->caps);
    }
    ucs_array_cleanup_dynamic(&proto_init->priv_buf);
    ucs_array_cleanup_dynamic(&proto_init->protocols);
}

static const char *ucp_proto_select_node_name(ucp_proto_perf_type_t perf_type)
{
    switch (perf_type) {
    case UCP_PROTO_PERF_TYPE_SINGLE:
        return "best single operation";
    case UCP_PROTO_PERF_TYPE_MULTI:
        return "best multiple operations";
    default:
        ucs_fatal("invalid performance type %d", perf_type);
    }
}

static ucs_status_t ucp_proto_select_elem_add_envelope(
        const ucp_proto_select_init_protocols_t *proto_init,
        ucp_worker_cfg_index_t ep_cfg_index,
        ucp_worker_cfg_index_t rkey_cfg_index,
        const ucp_proto_select_param_t *select_param, size_t msg_length,
        const ucp_proto_perf_envelope_t *envelope,
        const ucs_dynamic_bitmap_t *proto_mask, ucp_proto_thresh_t *thresholds,
        unsigned *last_proto_idx, ucp_proto_ranges_t *perf_ranges)
{
    const ucp_proto_perf_range_t *caps_range, *child_range;
    ucp_proto_perf_envelope_elem_t *envelope_elem;
    ucp_proto_threshold_elem_t *thresh_elem;
    unsigned proto_idx, child_proto_idx;
    const ucp_proto_init_elem_t *proto;
    ucp_proto_config_t *proto_config;
    ucp_proto_perf_type_t perf_type;
    ucp_proto_perf_range_t *range;
    const void *proto_priv;
    const char *node_name;
    size_t range_start;
    char range_str[64];

    perf_type = ucp_proto_select_param_perf_type(select_param);
    node_name = ucp_proto_select_node_name(perf_type);

    range_start = msg_length;
    ucs_array_for_each(envelope_elem, envelope) {
        proto_idx  = ucs_dynamic_bitmap_fns(proto_mask, envelope_elem->index);
        proto      = &ucs_array_elem(&proto_init->protocols, proto_idx);
        proto_priv = ucp_proto_select_init_priv_buf(proto_init, proto_idx);

        ucs_trace("%zu..%zu: %s", range_start, envelope_elem->max_length,
                  ucp_proto_id_field(proto->proto_id, name));

        if (*last_proto_idx == proto_idx) {
            /* If the last element used the same protocol - extend it */
            thresh_elem = ucs_array_last(thresholds);
            ucs_assertv(thresh_elem->proto_config.proto ==
                                ucp_protocols[proto->proto_id],
                        "thresh_elem->proto=%p proto=%p",
                        thresh_elem->proto_config.proto,
                        ucp_protocols[proto->proto_id]);
            ucs_assertv(thresh_elem->proto_config.priv == proto_priv,
                        "thresh_elem->priv=%p proto_priv=%p",
                        thresh_elem->proto_config.priv, proto_priv);
            thresh_elem->max_msg_length = envelope_elem->max_length;
        } else {
            thresh_elem = ucs_array_append(thresholds,
                                           return UCS_ERR_NO_MEMORY);

            thresh_elem->max_msg_length  = envelope_elem->max_length;
            proto_config                 = &thresh_elem->proto_config;
            proto_config->proto          = ucp_protocols[proto->proto_id];
            proto_config->priv           = proto_priv;
            proto_config->cfg_thresh     = proto->cfg_thresh;
            proto_config->ep_cfg_index   = ep_cfg_index;
            proto_config->rkey_cfg_index = rkey_cfg_index;
            proto_config->select_param   = *select_param;
            *last_proto_idx              = proto_idx;
        }

        /* Do not unite performance ranges, since they could have same final
         * result but calculated differently. And we want to track the whole
         * calculation process.
         */
        range = ucs_array_append(perf_ranges, return UCS_ERR_NO_MEMORY);

        caps_range = ucp_proto_caps_range_find(&proto->caps, range_start);
        ucs_assert(caps_range != NULL); /* for cppcheck */

        range->max_length = envelope_elem->max_length;
        ucp_proto_perf_copy(range->perf, caps_range->perf);

        ucs_memunits_range_str(range_start, envelope_elem->max_length,
                               range_str, sizeof(range_str));

        range->node = ucp_proto_perf_node_new_select(
                node_name, envelope_elem->index, "%s %s",
                ucp_proto_id_field(proto->proto_id, name), range_str);

        /* Add all candidates as children */
        UCS_DYNAMIC_BITMAP_FOR_EACH_BIT(child_proto_idx, proto_mask) {
            proto       = &ucs_array_elem(&proto_init->protocols,
                                          child_proto_idx);
            child_range = ucp_proto_caps_range_find(&proto->caps, range_start);
            ucp_proto_perf_node_add_child(range->node, child_range->node);
        }

        range_start = envelope_elem->max_length + 1;
    }

    return UCS_OK;
}

static ucs_status_t
ucp_proto_select_elem_init_thresh(ucp_worker_h worker,
                                  ucp_proto_select_elem_t *select_elem,
                                  ucp_proto_select_init_protocols_t *proto_init,
                                  ucp_worker_cfg_index_t ep_cfg_index,
                                  ucp_worker_cfg_index_t rkey_cfg_index,
                                  const ucp_proto_select_param_t *select_param)
{
    ucp_proto_thresh_t thresholds  = UCS_ARRAY_DYNAMIC_INITIALIZER;
    ucp_proto_ranges_t perf_ranges = UCS_ARRAY_DYNAMIC_INITIALIZER;
    unsigned last_proto_idx        = UINT_MAX;
    ucp_proto_perf_envelope_t envelope;
    ucp_proto_perf_list_t perf_list;
    ucs_dynamic_bitmap_t proto_mask;
    size_t msg_length, max_length;
    ucs_status_t status;

    ucs_dynamic_bitmap_init(&proto_mask);

    /*
     * Select a protocol for every message size interval, until we cover all
     * possible message sizes until SIZE_MAX.
     */
    msg_length = 0;
    do {
        ucs_array_init_dynamic(&perf_list);
        ucs_array_init_dynamic(&envelope);

        status = ucp_proto_thresholds_next_range(proto_init, select_param,
                                                 msg_length, &perf_list,
                                                 &max_length, &proto_mask);
        if (status != UCS_OK) {
            if (status == UCS_ERR_UNSUPPORTED) {
                ucs_debug("no protocol for msg_length %zu", msg_length);
            }
            goto err;
        }

        ucs_assert_always(!ucs_array_is_empty(&perf_list));

        status = ucp_proto_perf_envelope_make(&perf_list, msg_length,
                                              max_length, 1, &envelope);
        if (status != UCS_OK) {
            goto err_cleanup_perf_list;
        }

        ucs_assert_always(ucs_array_last(&envelope)->max_length == max_length);

        status = ucp_proto_select_elem_add_envelope(
                proto_init, ep_cfg_index, rkey_cfg_index, select_param,
                msg_length, &envelope, &proto_mask, &thresholds,
                &last_proto_idx, &perf_ranges);
        if (status != UCS_OK) {
            goto err_cleanup_envelope;
        }

        ucs_array_cleanup_dynamic(&envelope);
        ucs_array_cleanup_dynamic(&perf_list);

        msg_length = max_length + 1;
    } while (max_length < SIZE_MAX);

    ucs_dynamic_bitmap_cleanup(&proto_mask);

    ucs_assert_always(!ucs_array_is_empty(&thresholds));

    /* Set pointer to priv buffer (to release it during cleanup) */
    select_elem->priv_buf    = ucs_array_extract_buffer(&proto_init->priv_buf);
    select_elem->perf_ranges = ucs_array_extract_buffer(&perf_ranges);
    select_elem->thresholds  = ucs_array_extract_buffer(&thresholds);

    return UCS_OK;

err_cleanup_envelope:
    ucs_array_cleanup_dynamic(&envelope);
err_cleanup_perf_list:
    ucs_array_cleanup_dynamic(&perf_list);
err:
    ucp_proto_select_perf_ranges_cleanup(ucs_array_begin(&perf_ranges),
                                         ucs_array_length(&perf_ranges));
    ucs_array_cleanup_dynamic(&perf_ranges);
    ucs_array_cleanup_dynamic(&thresholds);
    ucs_dynamic_bitmap_cleanup(&proto_mask);
    return status;
}

/**
 * Get map of lanes used in the selected protocols.
 */
static ucp_lane_map_t
ucp_proto_select_get_lane_map(ucp_worker_h worker,
                              const ucp_proto_select_elem_t *select_elem)
{
    ucp_lane_map_t lane_map = 0;
    size_t range_start, range_end;
    ucp_proto_query_attr_t query_attr;

    range_end = -1;
    do {
        range_start = range_end + 1;
        ucp_proto_select_elem_query(worker, select_elem, range_start,
                                    &query_attr);

        range_end = query_attr.max_msg_length;
        lane_map |= query_attr.lane_map;
    } while (range_end != SIZE_MAX);

    return lane_map;
}

/**
 * Activate UCP worker interfaces corresponding to the resources of lanes used
 * in the selected protocols.
 */
static void
ucp_proto_select_wiface_activate(ucp_worker_h worker,
                                 const ucp_proto_select_elem_t *select_elem,
                                 ucp_worker_cfg_index_t ep_cfg_index)
{
    ucp_ep_config_t *ep_config;
    ucp_lane_map_t lane_map;

    ep_config = ucp_worker_ep_config(worker, ep_cfg_index);
    lane_map  = ucp_proto_select_get_lane_map(worker, select_elem) &
                ~ep_config->proto_lane_map;
    ucp_wiface_process_for_each_lane(worker, ep_config, lane_map,
                                     ucp_worker_iface_progress_ep);

    ep_config->proto_lane_map |= lane_map;
}

static ucs_status_t
ucp_proto_select_elem_init(ucp_worker_h worker, int internal,
                           ucp_worker_cfg_index_t ep_cfg_index,
                           ucp_worker_cfg_index_t rkey_cfg_index,
                           const ucp_proto_select_param_t *select_param,
                           ucp_proto_select_elem_t *select_elem)
{
    UCS_STRING_BUFFER_ONSTACK(sel_param_strb, UCP_PROTO_SELECT_PARAM_STR_MAX);
    UCS_STRING_BUFFER_ONSTACK(config_name_strb, UCP_PROTO_SELECT_PARAM_STR_MAX);
    ucp_proto_select_init_protocols_t proto_init;
    ucs_status_t status;

    ucp_proto_select_info_str(worker, rkey_cfg_index, select_param,
                              ucp_operation_names, &sel_param_strb);
    ucp_ep_config_name(worker, ep_cfg_index, &config_name_strb);

    ucs_trace("worker %p: select protocols %s rkey[%d] for %s", worker,
              ucs_string_buffer_cstr(&config_name_strb), rkey_cfg_index,
              ucs_string_buffer_cstr(&sel_param_strb));

    ucs_log_indent(1);

    status = ucp_proto_select_init_protocols(worker, ep_cfg_index,
                                             rkey_cfg_index, select_param,
                                             &proto_init);
    if (status != UCS_OK) {
        goto out;
    }

    status = ucp_proto_select_elem_init_thresh(worker, select_elem, &proto_init,
                                               ep_cfg_index, rkey_cfg_index,
                                               select_param);
    if (status != UCS_OK) {
        goto out_cleanup_proto_init;
    }

    ucp_proto_select_wiface_activate(worker, select_elem, ep_cfg_index);

    if (!internal) {
        ucp_proto_select_elem_trace(worker, ep_cfg_index, rkey_cfg_index,
                                    select_param, select_elem);
    }

    status = UCS_OK;

out_cleanup_proto_init:
    ucp_proto_select_cleanup_protocols(&proto_init);
out:
    ucs_log_indent(-1);
    return status;
}

static void
ucp_proto_select_elem_cleanup(ucp_proto_select_elem_t *select_elem)
{
    ucp_proto_perf_range_t *range;

    range = select_elem->perf_ranges;
    do {
        ucp_proto_perf_node_deref(&range->node);
    } while ((range++)->max_length < SIZE_MAX);
    ucs_free(select_elem->perf_ranges);
    ucs_free((void*)select_elem->thresholds);
    ucs_free(select_elem->priv_buf);
}

static void ucp_proto_select_cache_reset(ucp_proto_select_t *proto_select)
{
    proto_select->cache.key   = UINT64_MAX;
    proto_select->cache.value = NULL;
}

ucp_proto_select_elem_t *
ucp_proto_select_lookup_slow(ucp_worker_h worker,
                             ucp_proto_select_t *proto_select, int internal,
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
    khiter    = kh_get(ucp_proto_select_hash, proto_select->hash, key.u64);
    if (khiter != kh_end(proto_select->hash)) {
        select_elem = &kh_value(proto_select->hash, khiter);
        goto out;
    }

    status = ucp_proto_select_elem_init(worker, internal, ep_cfg_index,
                                        rkey_cfg_index, select_param,
                                        &tmp_select_elem);
    if (status != UCS_OK) {
        return NULL;
    }

    /* add to hash after initializing the temp element, since calling
     * ucp_proto_select_elem_init() can recursively modify the hash
     */
    khiter = kh_put(ucp_proto_select_hash, proto_select->hash, key.u64,
                    &khret);
    ucs_assert_always(khret == UCS_KH_PUT_BUCKET_EMPTY);

    select_elem  = &kh_value(proto_select->hash, khiter);
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
    proto_select->hash = kh_init(ucp_proto_select_hash);
    if (proto_select->hash == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    ucp_proto_select_cache_reset(proto_select);
    return UCS_OK;
}

void ucp_proto_select_cleanup(ucp_proto_select_t *proto_select)
{
    ucp_proto_select_elem_t select_elem;

    kh_foreach_value(proto_select->hash, select_elem,
         ucp_proto_select_elem_cleanup(&select_elem)
    )
    kh_destroy(ucp_proto_select_hash, proto_select->hash);
}

void ucp_proto_select_add_proto(const ucp_proto_init_params_t *init_params,
                                size_t cfg_thresh, unsigned cfg_priority,
                                const ucp_proto_caps_t *proto_caps,
                                const void *priv, size_t priv_size)
{
    ucp_proto_select_init_protocols_t *proto_init = init_params->ctx;
    ucp_proto_id_t proto_id                       = init_params->proto_id;
    ucp_proto_init_elem_t *init_elem;
    const char *proto_name;
    char min_length_str[64];
    char thresh_str[64];
    size_t priv_offset;

    proto_name = ucp_proto_id_field(proto_id, name);

    /* A successful protocol initialization must return non-empty range */
    ucs_assertv(proto_caps->min_length < SIZE_MAX, "proto=%s", proto_name);
    ucs_assertv(proto_caps->num_ranges > 0, "proto=%s", proto_name);

    if (init_params->ep_config_key->err_mode != UCP_ERR_HANDLING_MODE_NONE) {
        ucs_assertv(ucp_protocols[proto_id]->abort !=
                            ucp_proto_abort_fatal_not_implemented,
                    "error handling is enabled, but %s doesn't support abort()",
                    proto_name);
    }

    ucs_trace("added protocol %s min_length %s cfg_thresh %s cfg_priority %d "
              "priv_size %zu",
              init_params->proto_name,
              ucs_memunits_to_str(proto_caps->min_length, min_length_str,
                                  sizeof(min_length_str)),
              ucs_memunits_to_str(cfg_thresh, thresh_str, sizeof(thresh_str)),
              cfg_priority, priv_size);

    ucs_log_indent(1);
    ucp_proto_select_init_trace_caps(init_params);
    ucs_log_indent(-1);

    /* Copy private data */
    priv_offset = ucs_array_length(&proto_init->priv_buf);
    ucs_array_resize(&proto_init->priv_buf, priv_offset + priv_size, 0,
                     ucs_error("failed to allocate proto priv of size %zu",
                               priv_size);
                     return);
    memcpy(&ucs_array_elem(&proto_init->priv_buf, priv_offset), priv,
           priv_size);

    /* Add capabilities to the array of protocols */
    init_elem = ucs_array_append(
            &proto_init->protocols,
            ucs_error("failed to allocate protocol %s init element",
                      proto_name);
            /* Revert priv_buf length */
            ucs_array_set_length(&proto_init->priv_buf, priv_offset);
            return);

    memset(init_elem, 0, sizeof(*init_elem));
    init_elem->proto_id        = proto_id;
    init_elem->priv_offset     = priv_offset;
    init_elem->cfg_thresh      = cfg_thresh;
    init_elem->cfg_priority    = cfg_priority;
    init_elem->caps.min_length = proto_caps->min_length;
    init_elem->caps.num_ranges = proto_caps->num_ranges;
    memcpy(init_elem->caps.ranges, proto_caps->ranges,
           proto_caps->num_ranges * sizeof(*init_elem->caps.ranges));
}

void ucp_proto_select_short_disable(ucp_proto_select_short_t *proto_short)
{
    proto_short->max_length_unknown_mem = -1;
    proto_short->max_length_host_mem    = -1;
    proto_short->lane                   = UCP_NULL_LANE;
    proto_short->rkey_index             = UCP_NULL_RESOURCE;
}

void ucp_proto_select_short_init(ucp_worker_h worker,
                                 ucp_proto_select_t *proto_select,
                                 ucp_worker_cfg_index_t ep_cfg_index,
                                 ucp_worker_cfg_index_t rkey_cfg_index,
                                 ucp_operation_id_t op_id, unsigned proto_flags,
                                 ucp_proto_select_short_t *proto_short)
{
    static const uint32_t op_attributes[] = {0, UCP_OP_ATTR_FLAG_FAST_CMPL,
                                             UCP_OP_ATTR_FLAG_MULTI_SEND};
    ucp_context_h context                 = worker->context;
    const ucp_proto_t *proto              = NULL;
    const ucp_proto_threshold_elem_t *thresh;
    ucp_proto_select_param_t select_param;
    const ucp_proto_single_priv_t *spriv;
    ucp_memory_info_t mem_info;
    ssize_t max_short_signed;
    const uint32_t *op_attribute;

    ucp_memory_info_set_host(&mem_info);

    /*
     * Find the minimal threshold for operation 'op_id' among all protocols
     * with attribute from 'op_attributes'. Fast-path short protocol
     * can be used only if the message size fits this minimal threshold.
     */
    ucs_log_indent(1);
    ucs_carray_for_each(op_attribute, op_attributes,
                        ucs_static_array_size(op_attributes)) {
        ucp_proto_select_param_init(&select_param, op_id, *op_attribute, 0,
                                    UCP_DATATYPE_CONTIG, &mem_info, 1);
        thresh = ucp_proto_select_lookup(worker, proto_select, ep_cfg_index,
                                         rkey_cfg_index, &select_param, 0);
        if (thresh == NULL) {
            /* no protocol for contig/host */
            goto out_disable;
        }

        ucs_assert(thresh->proto_config.proto != NULL);
        if (!ucs_test_all_flags(thresh->proto_config.proto->flags,
                                proto_flags)) {
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

ucp_proto_select_t *
ucp_proto_select_get(ucp_worker_h worker, ucp_worker_cfg_index_t ep_cfg_index,
                     ucp_worker_cfg_index_t rkey_cfg_index,
                     ucp_worker_cfg_index_t *new_rkey_cfg_index)
{
    ucp_rkey_config_key_t rkey_config_key;
    ucs_status_t status;

    if (rkey_cfg_index == UCP_WORKER_CFG_INDEX_NULL) {
        *new_rkey_cfg_index = UCP_WORKER_CFG_INDEX_NULL;
        return &ucs_array_elem(&worker->ep_config, ep_cfg_index).proto_select;
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

void ucp_proto_config_query(ucp_worker_h worker,
                            const ucp_proto_config_t *proto_config,
                            size_t msg_length,
                            ucp_proto_query_attr_t *proto_attr)
{
    ucp_proto_query_params_t params = {
        .proto         = proto_config->proto,
        .priv          = proto_config->priv,
        .worker        = worker,
        .select_param  = &proto_config->select_param,
        .ep_config_key = &ucs_array_elem(&worker->ep_config,
                                         proto_config->ep_cfg_index).key,
        .msg_length    = msg_length
    };

    proto_config->proto->query(&params, proto_attr);
}

int ucp_proto_select_elem_query(ucp_worker_h worker,
                                const ucp_proto_select_elem_t *select_elem,
                                size_t msg_length,
                                ucp_proto_query_attr_t *proto_attr)
{
    const ucp_proto_threshold_elem_t *thresh_elem;
    const ucp_proto_config_t *proto_config;

    thresh_elem  = ucp_proto_select_thresholds_search(select_elem, msg_length);
    proto_config = &thresh_elem->proto_config;

    ucp_proto_config_query(worker, proto_config, msg_length, proto_attr);

    proto_attr->max_msg_length = ucs_min(proto_attr->max_msg_length,
                                         thresh_elem->max_msg_length);

    return !(thresh_elem->proto_config.proto->flags & UCP_PROTO_FLAG_INVALID);
}
