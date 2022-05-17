/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
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

#include <ucp/core/ucp_worker.inl>
#include <ucs/datastruct/array.inl>


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


UCS_ARRAY_DEFINE_INLINE(ucp_proto_ranges, unsigned, ucp_proto_select_range_t);
UCS_ARRAY_DEFINE_INLINE(ucp_proto_thresh, unsigned, ucp_proto_threshold_elem_t);


const ucp_proto_threshold_elem_t*
ucp_proto_thresholds_search_slow(const ucp_proto_threshold_elem_t *thresholds,
                                 size_t msg_length)
{
    unsigned idx;
    for (idx = 0; msg_length > thresholds[idx].max_msg_length; ++idx);
    return &thresholds[idx];
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


/*
 * Select a protocol for 'msg_length', return last message length for the proto
 */
static ucs_status_t ucp_proto_thresholds_select_next(
        ucp_proto_id_mask_t proto_mask, const ucp_proto_caps_t *proto_caps,
        ucp_proto_perf_type_t perf_type,
        ucp_proto_perf_envelope_t *thresh_list,
        size_t msg_length, size_t *max_length_p)
{
    ucp_proto_perf_envelope_t proto_list = UCS_ARRAY_DYNAMIC_INITIALIZER;
    ucp_proto_id_mask_t valid_proto_mask, disabled_proto_mask;
    ucp_proto_perf_range_t proto_perf[UCP_PROTO_MAX_COUNT];
    char range_str[64], time_str[64], bw_str[64];
    ucp_proto_perf_envelope_elem_t *elem;
    ucp_proto_id_t max_prio_proto_id;
    const ucp_proto_caps_t *caps;
    unsigned max_cfg_priority;
    ucp_proto_id_t proto_id;
    ucs_status_t status;
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
        ucs_assert(proto_id < ucp_protocols_count());
        /* coverity[overrun-local] */
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

    ucs_trace("  %-20s %-20s %-18s", "PROTOCOL", "TIME (ns/KB)", "BANDWIDTH (MB/s)");
    ucs_for_each_bit(proto_id, valid_proto_mask) {
        ucp_proto_select_perf_str(&proto_perf[proto_id].perf[perf_type],
                                  time_str, sizeof(time_str), bw_str,
                                  sizeof(bw_str));
        ucs_trace("  %-20s %-20s %-18s",
                  ucp_proto_id_field(proto_id, name), time_str, bw_str);

        status = ucs_array_append(ucp_proto_perf_envelope, &proto_list);
        if (status != UCS_OK) {
            goto out;
        }

        elem                   = ucs_array_last(&proto_list);
        elem->id               = proto_id;
        elem->range.super      = proto_perf[proto_id];
        elem->range.super.name = ucp_proto_id_field(proto_id, name);
        elem->range.cfg_thresh = proto_caps[proto_id].cfg_thresh;
    }

    status = ucp_proto_perf_envelope_make(&proto_list, thresh_list, msg_length,
                                          max_length, perf_type, 1);

    if (status == UCS_OK) {
        *max_length_p = max_length;
    }

out:
    ucs_log_indent(-1);
    ucs_array_cleanup_dynamic(&proto_list);
    return status;
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
    proto_init->priv_buf = ucs_malloc(ucp_protocols_count() * UCP_PROTO_PRIV_MAX,
                                      "ucp_proto_priv");
    if (proto_init->priv_buf == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    offset = 0;
    ucs_for_each_bit(proto_id, worker->context->proto_bitmap) {
        ucs_assert(proto_id < ucp_protocols_count());
        proto_caps             = &proto_init->caps[proto_id];
        init_params.priv       = UCS_PTR_BYTE_OFFSET(proto_init->priv_buf,
                                                     offset);
        init_params.priv_size  = &priv_size;
        init_params.caps       = proto_caps;
        /* coverity[overrun-local] */
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
        ucp_proto_select_param_str(select_param, ucp_operation_names, &strb);
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
    ucp_proto_perf_envelope_t envelope        = UCS_ARRAY_DYNAMIC_INITIALIZER;
    ucs_array_t(ucp_proto_ranges) perf_ranges = UCS_ARRAY_DYNAMIC_INITIALIZER;
    ucs_array_t(ucp_proto_thresh) thresholds  = UCS_ARRAY_DYNAMIC_INITIALIZER;
    ucp_proto_threshold_elem_t *thresh_elem   = NULL;
    ucp_proto_select_range_t *range_elem      = NULL;
    ucp_proto_perf_envelope_elem_t *envelope_elem;
    ucp_proto_config_t *proto_config;
    size_t msg_length, max_length;
    ucp_proto_id_t proto_id;
    ucs_status_t status;
    size_t priv_offset;

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
                                                  &envelope,
                                                  msg_length, &max_length);
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

    ucs_assert_always(!ucs_array_is_empty(&envelope));
    ucs_assert_always(ucs_array_last(&envelope)->range.super.max_length ==
                      SIZE_MAX);

    /* Copy the temporary thresholds list to an array inside select_elem */
    ucs_array_for_each(envelope_elem, &envelope) {
        proto_id    = envelope_elem->id;
        priv_offset = proto_init->priv_offsets[proto_id];

        if ((thresh_elem != NULL) &&
            (thresh_elem->proto_config.proto == ucp_protocols[proto_id])) {
            /* If the last element used the same protocol - extend it */
            thresh_elem->max_msg_length
                = envelope_elem->range.super.max_length;
        } else {
            status = ucs_array_append(ucp_proto_thresh, &thresholds);
            if (status != UCS_OK) {
                goto err;
            }

            thresh_elem                  = ucs_array_last(&thresholds);
            thresh_elem->max_msg_length
                = envelope_elem->range.super.max_length;
            proto_config                 = &thresh_elem->proto_config;
            proto_config->select_param   = *proto_init->select_param;
            proto_config->ep_cfg_index   = ep_cfg_index;
            proto_config->rkey_cfg_index = rkey_cfg_index;
            proto_config->proto          = ucp_protocols[proto_id];
            proto_config->priv
                = UCS_PTR_BYTE_OFFSET(select_elem->priv_buf, priv_offset);
        }

        if ((range_elem != NULL) &&
            ucp_proto_select_range_is_equal(range_elem,
                                            &envelope_elem->range)) {
            /* If the last range had the same performance - extend it */
            range_elem->super.max_length
                = envelope_elem->range.super.max_length;
        } else {
            status = ucs_array_append(ucp_proto_ranges, &perf_ranges);
            if (status != UCS_OK) {
                goto err;
            }

            range_elem  = ucs_array_last(&perf_ranges);
            *range_elem = envelope_elem->range;
        }
    }

    ucs_assert_always(!ucs_array_is_empty(&thresholds));

    select_elem->perf_ranges = ucs_array_extract_buffer(ucp_proto_ranges,
                                                        &perf_ranges);
    select_elem->thresholds  = ucs_array_extract_buffer(ucp_proto_thresh,
                                                        &thresholds);
    status = UCS_OK;

err:
    ucs_array_cleanup_dynamic(&envelope);
    ucs_array_cleanup_dynamic(&perf_ranges);
    ucs_array_cleanup_dynamic(&thresholds);
    return status;
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

    ucp_proto_select_param_str(select_param, ucp_operation_names,
                               &sel_param_strb);
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
        ucs_free(proto_init->priv_buf);
        goto out_free_proto_init;
    }

    ucp_proto_select_elem_trace(worker, ep_cfg_index, rkey_cfg_index,
                                select_param, select_elem);
    status = UCS_OK;

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
        ucp_proto_select_param_init(&select_param, op_id, op_attr, 0,
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
        .ep_config_key = &worker->ep_config[proto_config->ep_cfg_index].key,
        .msg_length    = msg_length
    };

    proto_config->proto->query(&params, proto_attr);
}

int ucp_proto_select_elem_query(ucp_worker_h worker,
                                const ucp_proto_select_elem_t *select_elem,
                                size_t msg_length,
                                ucp_proto_query_attr_t *proto_attr)
{
    const ucp_proto_threshold_elem_t *thresh_elem =
            ucp_proto_select_thresholds_search(select_elem, msg_length);

    ucp_proto_config_query(worker, &thresh_elem->proto_config, msg_length,
                           proto_attr);

    proto_attr->max_msg_length = ucs_min(proto_attr->max_msg_length,
                                         thresh_elem->max_msg_length);

    return !(thresh_elem->proto_config.proto->flags & UCP_PROTO_FLAG_INVALID);
}
