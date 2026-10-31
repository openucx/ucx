/*
 * Copyright (C) MThreads 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "musa_copy_iface.h"
#include "musa_copy_ep.h"

#include <uct/musa/base/musa_base.h>

#include <ucs/sys/sys.h>
#include <ucs/type/class.h>

#include <limits.h>
#include <stdlib.h>
#include <string.h>

extern uct_component_t uct_musa_component;

static void UCS_CLASS_DELETE_FUNC_NAME(uct_musa_copy_iface_t)(uct_iface_t *);

static ucs_config_field_t uct_musa_copy_iface_config_table[] = {
    {"", "", NULL, ucs_offsetof(uct_musa_copy_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},

    {"MAX_POLL", "16", "Max number of event completions to poll",
     ucs_offsetof(uct_musa_copy_iface_config_t, max_poll),
     UCS_CONFIG_TYPE_UINT},

    {"MAX_EVENTS", "inf", "Max number of MUSA events",
     ucs_offsetof(uct_musa_copy_iface_config_t, max_events),
     UCS_CONFIG_TYPE_UINT},

    {"LAT", "8us", "Latency",
     ucs_offsetof(uct_musa_copy_iface_config_t, latency), UCS_CONFIG_TYPE_TIME},

    {"BW", "10000MBs", "Effective memory bandwidth",
     ucs_offsetof(uct_musa_copy_iface_config_t, bandwidth),
     UCS_CONFIG_TYPE_BW},

    {NULL}
};

static void
uct_musa_copy_event_desc_init(ucs_mpool_t *mp, void *obj, void *chunk)
{
    uct_musa_copy_event_desc_t *event_desc = obj;

    event_desc->event = NULL;
    event_desc->comp  = NULL;
    event_desc->flush = NULL;
}

static void
uct_musa_copy_event_desc_cleanup(ucs_mpool_t *mp, void *obj)
{
    uct_musa_copy_device_rsc_t *device_rsc =
            ucs_container_of(mp, uct_musa_copy_device_rsc_t, event_mp);
    uct_musa_copy_event_desc_t *event_desc = obj;
    int current_device                    = -1;
    int have_current_device;

    if (event_desc->event == NULL) {
        return;
    }

    have_current_device =
            (uct_musa_base_get_current_device(&current_device) == UCS_OK);
    if (uct_musa_base_set_device(device_rsc->device) != UCS_OK) {
        return;
    }

    if (uct_musa_base_status(musaEventDestroy(event_desc->event),
                             "musaEventDestroy") != UCS_OK) {
        ucs_warn("failed to destroy MUSA event on device %d",
                 device_rsc->device);
    }

    uct_musa_base_restore_device(current_device, have_current_device);
}

static ucs_mpool_ops_t uct_musa_copy_event_desc_mpool_ops = {
    .chunk_alloc   = ucs_mpool_chunk_malloc,
    .chunk_release = ucs_mpool_chunk_free,
    .obj_init      = uct_musa_copy_event_desc_init,
    .obj_cleanup   = uct_musa_copy_event_desc_cleanup,
    .obj_str       = NULL
};

static void
uct_musa_copy_queue_desc_init(uct_musa_copy_queue_desc_t *queue_desc)
{
    queue_desc->stream = NULL;
    ucs_queue_head_init(&queue_desc->event_queue);
}

uct_musa_copy_device_rsc_t *
uct_musa_copy_iface_get_device_rsc(uct_musa_copy_iface_t *iface, int device)
{
    if ((device < 0) || ((unsigned)device >= iface->num_devices)) {
        return NULL;
    }

    return &iface->device_rscs[device];
}

ucs_status_t
uct_musa_copy_iface_get_stream(uct_musa_copy_device_rsc_t *device_rsc,
                               uct_musa_copy_queue_desc_t *queue_desc)
{
    ucs_status_t status;
    int current_device = -1;
    int have_current_device;

    if (queue_desc->stream != NULL) {
        return UCS_OK;
    }

    have_current_device =
            (uct_musa_base_get_current_device(&current_device) == UCS_OK);
    status = uct_musa_base_set_device(device_rsc->device);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_musa_base_status(
            musaStreamCreateWithFlags(&queue_desc->stream,
                                      musaStreamNonBlocking),
            "musaStreamCreateWithFlags");
    uct_musa_base_restore_device(current_device, have_current_device);
    return status;
}

static ucs_status_t
uct_musa_copy_get_event(uct_musa_copy_device_rsc_t *device_rsc,
                        uct_musa_copy_event_desc_t **event_desc_p)
{
    uct_musa_copy_event_desc_t *event_desc;
    ucs_status_t status;
    int current_device = -1;
    int have_current_device;

    event_desc = ucs_mpool_get(&device_rsc->event_mp);
    if (event_desc == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    if (event_desc->event != NULL) {
        *event_desc_p = event_desc;
        return UCS_OK;
    }

    have_current_device =
            (uct_musa_base_get_current_device(&current_device) == UCS_OK);
    status = uct_musa_base_set_device(device_rsc->device);
    if (status != UCS_OK) {
        goto err_put;
    }

    status = uct_musa_base_status(
            musaEventCreateWithFlags(&event_desc->event,
                                     musaEventDisableTiming),
            "musaEventCreateWithFlags");
    uct_musa_base_restore_device(current_device, have_current_device);
    if (status != UCS_OK) {
        goto err_put;
    }

    *event_desc_p = event_desc;
    return UCS_OK;

err_put:
    ucs_mpool_put(event_desc);
    return status;
}

ucs_status_t
uct_musa_copy_iface_post_event(uct_musa_copy_device_rsc_t *device_rsc,
                               uct_musa_copy_queue_desc_t *queue_desc,
                               uct_completion_t *comp,
                               uct_musa_copy_flush_desc_t *flush)
{
    uct_musa_copy_event_desc_t *event_desc;
    ucs_status_t status;
    int current_device = -1;
    int have_current_device;

    status = uct_musa_copy_iface_get_stream(device_rsc, queue_desc);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_musa_copy_get_event(device_rsc, &event_desc);
    if (status != UCS_OK) {
        return status;
    }

    have_current_device =
            (uct_musa_base_get_current_device(&current_device) == UCS_OK);
    status = uct_musa_base_set_device(device_rsc->device);
    if (status == UCS_OK) {
        status = uct_musa_base_status(
                musaEventRecord(event_desc->event, queue_desc->stream),
                "musaEventRecord");
    }
    uct_musa_base_restore_device(current_device, have_current_device);
    if (status != UCS_OK) {
        ucs_mpool_put(event_desc);
        return status;
    }

    if (ucs_queue_is_empty(&queue_desc->event_queue)) {
        ucs_queue_push(&device_rsc->active_queue, &queue_desc->queue);
    }

    event_desc->comp  = comp;
    event_desc->flush = flush;
    ucs_queue_push(&queue_desc->event_queue, &event_desc->queue);
    return UCS_INPROGRESS;
}

static void
uct_musa_copy_complete_event(uct_musa_copy_event_desc_t *event_desc,
                             ucs_status_t status)
{
    uct_musa_copy_flush_desc_t *flush;

    flush = event_desc->flush;
    if (flush == NULL) {
        if (event_desc->comp != NULL) {
            uct_invoke_completion(event_desc->comp, status);
        }
        return;
    }

    if (status != UCS_OK) {
        flush->status = status;
    }

    if (--flush->count == 0) {
        if (flush->comp != NULL) {
            uct_invoke_completion(flush->comp, flush->status);
        }
        ucs_free(flush);
    }
}

static unsigned
uct_musa_copy_progress_device(uct_musa_copy_iface_t *iface,
                              uct_musa_copy_device_rsc_t *device_rsc)
{
    uct_musa_copy_event_desc_t *event_desc;
    uct_musa_copy_queue_desc_t *queue_desc;
    ucs_queue_iter_t queue_iter;
    ucs_queue_iter_t event_iter;
    ucs_status_t status;
    musaError_t error;
    unsigned count = 0;

    ucs_queue_for_each_safe(queue_desc, queue_iter,
                            &device_rsc->active_queue, queue) {
        ucs_queue_for_each_safe(event_desc, event_iter,
                                &queue_desc->event_queue, queue) {
            if (count >= iface->config.max_poll) {
                return count;
            }

            error = musaEventQuery(event_desc->event);
            if (error == musaErrorNotReady) {
                break;
            }

            ucs_queue_del_iter(&queue_desc->event_queue, event_iter);
            status = uct_musa_base_status(error, "musaEventQuery");
            uct_musa_copy_complete_event(event_desc, status);
            ucs_mpool_put(event_desc);
            ++count;
        }

        if (ucs_queue_is_empty(&queue_desc->event_queue)) {
            ucs_queue_del_iter(&device_rsc->active_queue, queue_iter);
        }
    }

    return count;
}

unsigned uct_musa_copy_iface_progress(uct_iface_h tl_iface)
{
    uct_musa_copy_iface_t *iface = ucs_derived_of(tl_iface,
                                                  uct_musa_copy_iface_t);
    uct_musa_copy_device_rsc_t *device_rsc;
    unsigned count = 0;
    unsigned i;
    int current_device = -1;
    int have_current_device;

    for (i = 0; i < iface->num_devices; ++i) {
        device_rsc = &iface->device_rscs[i];
        if (ucs_queue_is_empty(&device_rsc->active_queue)) {
            continue;
        }

        have_current_device =
                (uct_musa_base_get_current_device(&current_device) == UCS_OK);
        if (uct_musa_base_set_device(device_rsc->device) != UCS_OK) {
            continue;
        }

        count += uct_musa_copy_progress_device(iface, device_rsc);
        uct_musa_base_restore_device(current_device, have_current_device);
        if (count >= iface->config.max_poll) {
            break;
        }
    }

    return count;
}

ucs_status_t
uct_musa_copy_iface_flush(uct_iface_h tl_iface, unsigned flags,
                          uct_completion_t *comp)
{
    uct_musa_copy_iface_t *iface = ucs_derived_of(tl_iface,
                                                  uct_musa_copy_iface_t);
    uct_musa_copy_device_rsc_t *device_rsc;
    uct_musa_copy_queue_desc_t *queue_desc;
    uct_musa_copy_flush_desc_t *flush;
    ucs_queue_iter_t queue_iter;
    ucs_status_t status;
    unsigned i;

    for (i = 0; i < iface->num_devices; ++i) {
        if (!ucs_queue_is_empty(&iface->device_rscs[i].active_queue)) {
            break;
        }
    }
    if (i == iface->num_devices) {
        UCT_TL_IFACE_STAT_FLUSH(&iface->super);
        return UCS_OK;
    }

    if (comp == NULL) {
        UCT_TL_IFACE_STAT_FLUSH_WAIT(&iface->super);
        return UCS_INPROGRESS;
    }

    flush = ucs_malloc(sizeof(*flush), "musa_copy_flush");
    if (flush == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    flush->count  = 0;
    flush->comp   = comp;
    flush->status = UCS_OK;

    for (i = 0; i < iface->num_devices; ++i) {
        device_rsc = &iface->device_rscs[i];
        ucs_queue_for_each_safe(queue_desc, queue_iter,
                                &device_rsc->active_queue, queue) {
            status = uct_musa_copy_iface_post_event(device_rsc, queue_desc,
                                                    NULL, flush);
            if (status != UCS_INPROGRESS) {
                flush->comp = NULL;
                if (flush->count == 0) {
                    ucs_free(flush);
                }
                return status;
            }
            ++flush->count;
        }
    }

    UCT_TL_IFACE_STAT_FLUSH_WAIT(&iface->super);
    return UCS_INPROGRESS;
}

static ucs_status_t
uct_musa_copy_iface_get_address(uct_iface_h tl_iface,
                                uct_iface_addr_t *iface_addr)
{
    uct_musa_copy_iface_t *iface = ucs_derived_of(tl_iface,
                                                  uct_musa_copy_iface_t);

    *(uct_musa_copy_iface_addr_t*)iface_addr = iface->id;
    return UCS_OK;
}

static int
uct_musa_copy_iface_is_reachable_v2(
        const uct_iface_h tl_iface,
        const uct_iface_is_reachable_params_t *params)
{
    uct_musa_copy_iface_t *iface = ucs_derived_of(tl_iface,
                                                  uct_musa_copy_iface_t);
    const uct_musa_copy_iface_addr_t *addr;

    if (!uct_iface_is_reachable_params_addrs_valid(params)) {
        return 0;
    }

    addr = (const uct_musa_copy_iface_addr_t*)params->iface_addr;
    if ((addr == NULL) || (*addr != iface->id)) {
        uct_iface_fill_info_str_buf(params, "iface address mismatch");
        return 0;
    }

    return uct_iface_scope_is_reachable(tl_iface, params);
}

static ucs_status_t
uct_musa_copy_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_musa_copy_iface_t *iface = ucs_derived_of(tl_iface,
                                                  uct_musa_copy_iface_t);

    uct_base_iface_query(&iface->super, iface_attr);

    iface_attr->iface_addr_len          = sizeof(uct_musa_copy_iface_addr_t);
    iface_attr->device_addr_len         = 0;
    iface_attr->ep_addr_len             = 0;
    iface_attr->cap.flags               = UCT_IFACE_FLAG_CONNECT_TO_IFACE |
                                          UCT_IFACE_FLAG_GET_SHORT |
                                          UCT_IFACE_FLAG_PUT_SHORT |
                                          UCT_IFACE_FLAG_GET_ZCOPY |
                                          UCT_IFACE_FLAG_PUT_ZCOPY |
                                          UCT_IFACE_FLAG_PENDING;
    iface_attr->cap.put.max_short       = UINT_MAX;
    iface_attr->cap.put.max_bcopy       = 0;
    iface_attr->cap.put.min_zcopy       = 0;
    iface_attr->cap.put.max_zcopy       = SIZE_MAX;
    iface_attr->cap.put.opt_zcopy_align = 1;
    iface_attr->cap.put.align_mtu       = 1;
    iface_attr->cap.put.max_iov         = 1;

    iface_attr->cap.get.max_short       = UINT_MAX;
    iface_attr->cap.get.max_bcopy       = 0;
    iface_attr->cap.get.min_zcopy       = 0;
    iface_attr->cap.get.max_zcopy       = SIZE_MAX;
    iface_attr->cap.get.opt_zcopy_align = 1;
    iface_attr->cap.get.align_mtu       = 1;
    iface_attr->cap.get.max_iov         = 1;

    iface_attr->cap.am.max_short        = 0;
    iface_attr->cap.am.max_bcopy        = 0;
    iface_attr->cap.am.min_zcopy        = 0;
    iface_attr->cap.am.max_zcopy        = 0;
    iface_attr->cap.am.opt_zcopy_align  = 1;
    iface_attr->cap.am.align_mtu        = 1;
    iface_attr->cap.am.max_hdr          = 0;
    iface_attr->cap.am.max_iov          = 1;

    iface_attr->latency                 = ucs_linear_func_make(
            iface->config.latency, 0);
    iface_attr->bandwidth.dedicated     = 0;
    iface_attr->bandwidth.shared        = iface->config.bandwidth;
    iface_attr->overhead                = 0;
    iface_attr->priority                = 0;
    return UCS_OK;
}

static uct_iface_ops_t uct_musa_copy_iface_ops = {
    .ep_get_short             = uct_musa_copy_ep_get_short,
    .ep_put_short             = uct_musa_copy_ep_put_short,
    .ep_get_zcopy             = uct_musa_copy_ep_get_zcopy,
    .ep_put_zcopy             = uct_musa_copy_ep_put_zcopy,
    .ep_pending_add           = (uct_ep_pending_add_func_t)
                                ucs_empty_function_return_busy,
    .ep_pending_purge         = (uct_ep_pending_purge_func_t)
                                ucs_empty_function,
    .ep_flush                 = uct_musa_copy_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .ep_create                = UCS_CLASS_NEW_FUNC_NAME(uct_musa_copy_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_musa_copy_ep_t),
    .iface_flush              = uct_musa_copy_iface_flush,
    .iface_fence              = uct_base_iface_fence,
    .iface_progress_enable    = uct_base_iface_progress_enable,
    .iface_progress_disable   = uct_base_iface_progress_disable,
    .iface_progress           = uct_musa_copy_iface_progress,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(
                                    uct_musa_copy_iface_t),
    .iface_query              = uct_musa_copy_iface_query,
    .iface_get_device_address = (uct_iface_get_device_address_func_t)
                                ucs_empty_function_return_success,
    .iface_get_address        = uct_musa_copy_iface_get_address,
    .iface_is_reachable       = uct_base_iface_is_reachable
};

static uct_iface_internal_ops_t uct_musa_copy_iface_internal_ops = {
    .iface_query_v2        = uct_iface_base_query_v2,
    .iface_estimate_perf   = uct_base_iface_estimate_perf,
    .iface_vfs_refresh     = (uct_iface_vfs_refresh_func_t)ucs_empty_function,
    .ep_query              = (uct_ep_query_func_t)
                             ucs_empty_function_return_unsupported,
    .ep_invalidate         = (uct_ep_invalidate_func_t)
                             ucs_empty_function_return_unsupported,
    .ep_connect_to_ep_v2   = (uct_ep_connect_to_ep_v2_func_t)
                             ucs_empty_function_return_unsupported,
    .iface_is_reachable_v2 = uct_musa_copy_iface_is_reachable_v2,
    .ep_is_connected       = uct_base_ep_is_connected,
    .ep_get_device_ep      = (uct_ep_get_device_ep_func_t)
                             ucs_empty_function_return_unsupported,
    .ep_put_sgl_zcopy      = (uct_ep_put_sgl_zcopy_func_t)
                             ucs_empty_function_return_unsupported,
    .ep_get_sgl_zcopy      = (uct_ep_get_sgl_zcopy_func_t)
                             ucs_empty_function_return_unsupported,
    .ep_outstanding_purge  = (uct_ep_outstanding_purge_func_t)
                             ucs_empty_function_return_unsupported
};

static UCS_CLASS_INIT_FUNC(uct_musa_copy_iface_t, uct_md_h md,
                           uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_musa_copy_iface_config_t *config =
            ucs_derived_of(tl_config, uct_musa_copy_iface_config_t);
    const char *device_name = NULL;
    char *end;
    long device;
    unsigned device_count;
    unsigned i;
    unsigned j;
    unsigned k;
    ucs_mpool_params_t mp_params;
    ucs_status_t status;

    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &uct_musa_copy_iface_ops,
                              &uct_musa_copy_iface_internal_ops, md, worker,
                              params, tl_config
                              UCS_STATS_ARG(params->stats_root)
                              UCS_STATS_ARG(UCT_MUSA_COPY_TL_NAME));

    device = 0;
    if (params->field_mask & UCT_IFACE_PARAM_FIELD_DEVICE) {
        device_name = params->mode.device.dev_name;
        if ((device_name == NULL) ||
            (strncmp(device_name, "musa", 4) != 0)) {
            return UCS_ERR_NO_DEVICE;
        }

        device = strtol(device_name + 4, &end, 10);
        if ((end == (device_name + 4)) || (*end != '\0') || (device < 0)) {
            return UCS_ERR_INVALID_PARAM;
        }
    }

    status = uct_musa_base_get_device_count(&device_count);
    if ((status != UCS_OK) || ((unsigned)device >= device_count)) {
        return UCS_ERR_NO_DEVICE;
    }

    self->id               = ucs_generate_uuid((uintptr_t)self);
    self->device           = device;
    self->num_devices      = device_count;
    self->device_rscs      = ucs_calloc(device_count,
                                        sizeof(*self->device_rscs),
                                        "musa_copy_device_rscs");
    if (self->device_rscs == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    self->config.max_poll   = config->max_poll;
    self->config.max_events = config->max_events;
    self->config.latency    = config->latency;
    self->config.bandwidth  = config->bandwidth;

    for (i = 0; i < device_count; ++i) {
        self->device_rscs[i].device = i;
        ucs_queue_head_init(&self->device_rscs[i].active_queue);
        for (j = 0; j < UCS_MEMORY_TYPE_LAST; ++j) {
            for (k = 0; k < UCS_MEMORY_TYPE_LAST; ++k) {
                uct_musa_copy_queue_desc_init(
                        &self->device_rscs[i].queue_desc[j][k]);
            }
        }

        ucs_mpool_params_reset(&mp_params);
        mp_params.elem_size       = sizeof(uct_musa_copy_event_desc_t);
        mp_params.elems_per_chunk = 128;
        mp_params.max_elems       = config->max_events;
        mp_params.ops             = &uct_musa_copy_event_desc_mpool_ops;
        mp_params.name            = "musa_copy_event_descriptors";
        status = ucs_mpool_init(&mp_params, &self->device_rscs[i].event_mp);
        if (status != UCS_OK) {
            goto err_cleanup_resources;
        }
    }

    return UCS_OK;

err_cleanup_resources:
    while (i > 0) {
        --i;
        ucs_mpool_cleanup(&self->device_rscs[i].event_mp, 1);
    }
    ucs_free(self->device_rscs);
    self->device_rscs = NULL;
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_musa_copy_iface_t)
{
    uct_musa_copy_device_rsc_t *device_rsc;
    uct_musa_copy_queue_desc_t *queue_desc;
    unsigned i;
    unsigned j;
    unsigned k;
    int current_device = -1;
    int have_current_device;

    uct_base_iface_progress_disable(&self->super.super,
                                    UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);

    for (i = 0; i < self->num_devices; ++i) {
        device_rsc = &self->device_rscs[i];
        ucs_mpool_cleanup(&device_rsc->event_mp, 1);

        have_current_device =
                (uct_musa_base_get_current_device(&current_device) == UCS_OK);
        if (uct_musa_base_set_device(device_rsc->device) != UCS_OK) {
            continue;
        }

        for (j = 0; j < UCS_MEMORY_TYPE_LAST; ++j) {
            for (k = 0; k < UCS_MEMORY_TYPE_LAST; ++k) {
                queue_desc = &device_rsc->queue_desc[j][k];
                if (queue_desc->stream == NULL) {
                    continue;
                }

                if (!ucs_queue_is_empty(&queue_desc->event_queue)) {
                    ucs_warn("MUSA stream for device %d has %zu "
                             "outstanding events",
                             device_rsc->device,
                             ucs_queue_length(&queue_desc->event_queue));
                }

                if (uct_musa_base_status(
                            musaStreamDestroy(queue_desc->stream),
                            "musaStreamDestroy") != UCS_OK) {
                    ucs_warn("failed to destroy MUSA stream on device %d",
                             device_rsc->device);
                }
            }
        }

        uct_musa_base_restore_device(current_device, have_current_device);
    }

    ucs_free(self->device_rscs);
}

UCS_CLASS_DEFINE(uct_musa_copy_iface_t, uct_base_iface_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_musa_copy_iface_t, uct_iface_t, uct_md_h,
                          uct_worker_h, const uct_iface_params_t *,
                          const uct_iface_config_t *);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_musa_copy_iface_t, uct_iface_t);

static ucs_status_t
uct_musa_copy_query_devices(uct_md_h md,
                            uct_tl_device_resource_t **tl_devices_p,
                            unsigned *num_tl_devices_p)
{
    ucs_status_t status;
    unsigned full_count;

    status = uct_musa_base_query_devices(md, tl_devices_p, num_tl_devices_p);
    if (status != UCS_OK) {
        return status;
    }

    full_count = *num_tl_devices_p;

    /* FIXME: UCX memtype EP asserts num_lanes <= 2. Remove this cap when UCX
     * supports multi-lane memtype endpoints or selects a single device
     * deterministically (same limitation as ze_copy). The copy EP selects the
     * device from the pointer attributes, so a single iface is sufficient. */
    if (full_count > 1) {
        ucs_diag("musa_copy: %u devices found, limiting to 1 "
                 "for memtype ep compatibility",
                 full_count);
        *num_tl_devices_p = 1;
    }

    return UCS_OK;
}

UCT_TL_DEFINE(&uct_musa_component, musa_copy, uct_musa_copy_query_devices,
              uct_musa_copy_iface_t, "MUSA_COPY_",
              uct_musa_copy_iface_config_table, uct_musa_copy_iface_config_t);
