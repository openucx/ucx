/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "proto.h"
#include "proto_select.inl"

#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_worker.h>
#include <ucp/dt/dt.h>


/* Parameters structure for initializing protocols for a selection parameter */
typedef struct {
    const ucp_proto_select_param_t *select_param;
    ucp_proto_id_mask_t            mask;
    ucp_proto_caps_t               caps[UCP_PROTO_MAX_COUNT];
    void                           *priv_buf;
    size_t                         priv_offsets[UCP_PROTO_MAX_COUNT];
} ucp_proto_select_init_protocols_t;


static ucs_status_t
ucp_proto_select_init_protocols(ucp_worker_h worker,
                                ucp_worker_cfg_index_t ep_cfg_index,
                                ucp_worker_cfg_index_t rkey_cfg_index,
                                const ucp_proto_select_param_t *select_param,
                                ucp_proto_select_init_protocols_t *proto_init)
{
    ucp_proto_init_params_t init_params;
    ucs_string_buffer_t strb;
    size_t priv_size, offset;
    ucp_proto_id_t proto_id;
    ucs_status_t status;
    void *tmp;

    ucs_assert(ep_cfg_index != UCP_WORKER_CFG_INDEX_NULL);

    init_params.worker        = worker;
    init_params.select_param  = select_param;
    init_params.ep_config_key = &worker->ep_config[ep_cfg_index].key;

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
    for (proto_id = 0; proto_id < ucp_protocols_count; ++proto_id) {
        init_params.priv      = UCS_PTR_BYTE_OFFSET(proto_init->priv_buf,
                                                          offset);
        init_params.priv_size  = &priv_size;
        init_params.caps       = &proto_init->caps[proto_id];
        init_params.proto_name = ucp_proto_id_field(proto_id, name);

        status = ucp_proto_id_call(proto_id, init, &init_params);
        if (status != UCS_OK) {
            continue;
        }

        proto_init->mask                  |= UCS_BIT(proto_id);
        proto_init->priv_offsets[proto_id] = offset;
        offset                            += priv_size;
    }

    if (proto_init->mask == 0) {
        /* No protocol can support the given selection parameters */
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

static void
ucp_proto_select_elem_cleanup(ucp_proto_select_elem_t *select_elem)
{
    ucs_free(select_elem->thresholds);
    ucs_free(select_elem->priv_buf);
}

ucs_status_t ucp_proto_select_init(ucp_proto_select_t *proto_select)
{
    kh_init_inplace(ucp_proto_select_hash, &proto_select->hash);
    proto_select->cache.key   = UINT64_MAX;
    proto_select->cache.value = 0;
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

void ucp_proto_select_dump_all(ucp_worker_h worker,
                               ucp_worker_cfg_index_t ep_cfg_index,
                               ucp_worker_cfg_index_t rkey_cfg_index,
                               const ucp_proto_select_param_t *select_param,
                               FILE *stream)
{
    static const char *proto_info_fmt =
                                "#     %-18s %-12s %-20s %-18s %-12s %s\n";
    ucp_proto_select_init_protocols_t *proto_init;
    ucs_string_buffer_t config_strb;
    size_t range_start, range_end;
    const ucp_proto_caps_t *caps;
    ucp_proto_id_t proto_id;
    ucs_status_t status;
    char range_str[64];
    char perf_str[64];
    char thresh_str[64];
    char bw_str[64];
    unsigned i;
    void *priv;

    /* Allocate on heap, since the structure is quite large */
    proto_init = ucs_malloc(sizeof(*proto_init), "proto_init");
    if (proto_init == NULL) {
        fprintf(stream, "<Could not allocate memory>\n");
        return;
    }

    status = ucp_proto_select_init_protocols(worker, ep_cfg_index, rkey_cfg_index,
                                             select_param, proto_init);
    if (status != UCS_OK) {
        fprintf(stream, "<%s>\n", ucs_status_string(status));
        goto out_free;
    }

    fprintf(stream, proto_info_fmt, "PROTOCOL", "SIZE", "TIME (nsec)",
            "BANDWIDTH (MiB/s)", "THRESHOLD", "CONIFURATION");

    ucs_for_each_bit(proto_id, proto_init->mask) {

        priv = UCS_PTR_BYTE_OFFSET(proto_init->priv_buf,
                                   proto_init->priv_offsets[proto_id]);
        caps = &proto_init->caps[proto_id];

        /* Get protocol configuration */
        ucp_proto_id_call(proto_id, config_str, priv, &config_strb);

        /* String for configured threshold */
        ucs_memunits_to_str(caps->cfg_thresh, thresh_str, sizeof(thresh_str));

        range_start = caps->min_length;
        for (i = 0; i < caps->num_ranges; ++i) {
            /* String for performance range */
            range_end = caps->ranges[i].max_length;
            ucs_memunits_range_str(range_start, range_end, range_str,
                                   sizeof(range_str));

            /* String for estimated performance */
            snprintf(perf_str, sizeof(perf_str), "%5.0f + %.3f * N",
                     caps->ranges[i].perf.c * 1e9,
                     caps->ranges[i].perf.m * 1e9);

            /* String for bandwidth */
            snprintf(bw_str, sizeof(bw_str), "%7.2f",
                     1.0 / (caps->ranges[i].perf.m * UCS_MBYTE));

            fprintf(stream, proto_info_fmt,
                    (i == 0) ? ucp_proto_id_field(proto_id, name) : "",
                    range_str, perf_str, bw_str,
                    (i == 0) ? thresh_str : "",
                    (i == 0) ? ucs_string_buffer_cstr(&config_strb) : "");

            range_start = range_end + 1;
        }

        ucs_string_buffer_cleanup(&config_strb);
    }
    fprintf(stream, "#\n");

    ucs_free(proto_init->priv_buf);
out_free:
    ucs_free(proto_init);
}

static void
ucp_proto_select_dump_thresholds(const ucp_proto_select_elem_t *select_elem,
                                 FILE *stream)
{
    static const char *proto_info_fmt = "#     %-16s %-18s %s\n";
    const ucp_proto_threshold_elem_t *thresh_elem;
    ucs_string_buffer_t strb;
    size_t range_start, range_end;
    char str[128];

    range_start = 0;
    thresh_elem = select_elem->thresholds;
    fprintf(stream, proto_info_fmt, "SIZE", "PROTOCOL", "CONFIGURATION");
    do {
        thresh_elem->proto_config.proto->config_str(
                thresh_elem->proto_config.priv, &strb);

        range_end = thresh_elem->max_msg_length;

        fprintf(stream, proto_info_fmt,
                ucs_memunits_range_str(range_start, range_end, str, sizeof(str)),
                thresh_elem->proto_config.proto->name,
                ucs_string_buffer_cstr(&strb));

        ucs_string_buffer_cleanup(&strb);

        range_start = range_end + 1;
        ++thresh_elem;
    } while (range_end != SIZE_MAX);
}

static void
ucp_proto_select_elem_dump(ucp_worker_h worker,
                           ucp_worker_cfg_index_t ep_cfg_index,
                           ucp_worker_cfg_index_t rkey_cfg_index,
                           const ucp_proto_select_param_t *select_param,
                           const ucp_proto_select_elem_t *select_elem,
                           FILE *stream)
{
    ucs_string_buffer_t strb;
    size_t i;

    fprintf(stream, "#\n");

    ucp_proto_select_param_str(select_param, &strb);
    fprintf(stream, "# %s:\n", ucs_string_buffer_cstr(&strb));
    fprintf(stream, "# ");
    for (i = 0; i < strlen(ucs_string_buffer_cstr(&strb)); ++i) {
        fputc('=', stream);
    }
    fprintf(stream, "\n");
    ucs_string_buffer_cleanup(&strb);

    fprintf(stream, "#\n");
    fprintf(stream, "#   Selected protocols:\n");

    ucp_proto_select_dump_thresholds(select_elem, stream);

    fprintf(stream, "#\n");

    fprintf(stream, "#   Candidates:\n");
    ucp_proto_select_dump_all(worker, ep_cfg_index, rkey_cfg_index,
                              select_param, stream);
}

void ucp_proto_select_dump(ucp_worker_h worker,
                           ucp_worker_cfg_index_t ep_cfg_index,
                           ucp_worker_cfg_index_t rkey_cfg_index,
                           ucp_proto_select_t *proto_select, FILE *stream)
{
    ucp_proto_select_elem_t select_elem;
    ucp_proto_select_key_t key;

    fprintf(stream, "# \n");
    fprintf(stream, "# Protocols selection for ep_config[%d]/rkey_config[%d] "
            "(%d items)\n", ep_cfg_index, rkey_cfg_index,
            kh_size(&proto_select->hash));
    fprintf(stream, "# \n");
    kh_foreach(&proto_select->hash, key.u64, select_elem,
         ucp_proto_select_elem_dump(worker, ep_cfg_index, rkey_cfg_index,
                                    &key.param, &select_elem, stream);
    )
}

void ucp_proto_select_param_str(const ucp_proto_select_param_t *select_param,
                                ucs_string_buffer_t *strb)
{
    uint32_t op_attr_mask;

    ucs_string_buffer_init(strb);

    op_attr_mask = ucp_proto_select_op_attr_from_flags(select_param->op_flags);
    ucs_string_buffer_appendf(strb, "%s()",
                              ucp_operation_names[select_param->op_id]);
    ucs_string_buffer_appendf(strb, " on a %s data-type",
                              ucp_datatype_class_names[select_param->dt_class]);
    if (select_param->sg_count > 1) {
        ucs_string_buffer_appendf(strb, "with %u scatter-gather entries",
                                  select_param->sg_count);
    }
    ucs_string_buffer_appendf(strb, " in %s memory",
                              ucs_memory_type_names[select_param->mem_type]);

    if (op_attr_mask & UCP_OP_ATTR_FLAG_FAST_CMPL) {
        ucs_string_buffer_appendf(strb, " and fast completion");
    }
}
