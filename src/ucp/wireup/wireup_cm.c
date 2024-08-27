/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2019. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "wireup_cm.h"
#include <ucp/core/ucp_listener.h>
#include <ucp/core/ucp_request.inl>
#include <ucp/core/ucp_ep.h>
#include <ucp/wireup/wireup.h>
#include <ucp/wireup/wireup_ep.h>
#include <ucs/sys/sock.h>
#include <ucs/sys/string.h>


/**
 * @brief Check whether CM callback should be called or not.
 *
 * @param [in] _ucp_ep        UCP Endpoint for which CM callback is called.
 * @param [in] _uct_cm_ep     UCT CM Endpoint which calls CM callback.
 * @param [in] _failed_action Action to do if UCP EP is in a FAILED state.
 *                            This actions should stop macro execution.
 */
#define UCP_EP_CM_CALLBACK_ENTER(_ucp_ep, _uct_cm_ep, _failed_action) \
    do { \
        ucs_assert(ucs_async_is_blocked(&(_ucp_ep)->worker->async)); \
        if ((_ucp_ep)->flags & UCP_EP_FLAG_FAILED) { \
            _failed_action; \
        } \
        \
        ucs_assertv_always((_uct_cm_ep) == ucp_ep_get_cm_uct_ep(_ucp_ep), \
                           "%p: uct_cm_ep=%p vs found_uct_ep=%p", \
                           _ucp_ep, _uct_cm_ep, \
                           ucp_ep_get_cm_uct_ep(_ucp_ep)); \
    } while (0)


int ucp_ep_init_flags_has_cm(unsigned ep_init_flags)
{
    return !!(ep_init_flags & (UCP_EP_INIT_CM_WIREUP_CLIENT |
                               UCP_EP_INIT_CM_WIREUP_SERVER));
}

/*
 * The main thread progress part of attempting connecting the client to the server
 * through the next available cm.
 */
unsigned ucp_cm_client_try_next_cm_progress(void *arg)
{
    ucp_ep_h ucp_ep       = arg;
    ucp_worker_h worker   = ucp_ep->worker;
    ucp_context_h context = worker->context;
    ucp_wireup_ep_t *cm_wireup_ep;
    ucs_status_t status;
    ucp_rsc_index_t cm_idx;

    UCS_ASYNC_BLOCK(&worker->async);

    ucs_assert(!(ucp_ep->flags & UCP_EP_FLAG_FAILED));

    cm_idx = ucp_ep->ext->cm_idx;
    ucs_assert(cm_idx != UCP_NULL_RESOURCE);

    cm_wireup_ep = ucp_ep_get_cm_wireup_ep(ucp_ep);
    ucs_assert_always(cm_wireup_ep != NULL);
    if (ucp_wireup_ep_has_next_ep(cm_wireup_ep)) {
        /* If we failed to create CM ep, next_ep is not initialized */
        ucp_wireup_ep_destroy_next_ep(cm_wireup_ep);
    }

    ucs_diag("ep %p: client switching from %s to %s in attempt to connect to"
             " the server", ucp_ep, ucp_context_cm_name(context, cm_idx - 1),
             ucp_context_cm_name(context, cm_idx));

    status = ucp_ep_client_cm_create_uct_ep(ucp_ep);
    if (status != UCS_OK) {
        ucs_error("failed to create a uct sockaddr endpoint on %s cm %p",
                  ucp_context_cm_name(context, cm_idx), worker->cms[cm_idx].cm);

        ucp_ep_set_failed(ucp_ep, ucp_ep_get_cm_lane(ucp_ep), status);
    }

    UCS_ASYNC_UNBLOCK(&worker->async);
    return 1;
}

static int ucp_cm_client_get_next_cm_idx(ucp_ep_h ep)
{
    ucp_worker_h worker          = ep->worker;
    ucp_rsc_index_t next_cm_idx  = ep->ext->cm_idx + 1;
    ucp_rsc_index_t num_cm_cmpts = ucp_worker_num_cm_cmpts(worker);

    for (; next_cm_idx < num_cm_cmpts; ++next_cm_idx) {
        if (worker->cms[next_cm_idx].cm != NULL) {
            return next_cm_idx;
        }
    }

    return UCP_NULL_RESOURCE;
}

static int ucp_cm_client_try_fallback_cms(ucp_ep_h ep)
{
    ucp_worker_h worker          = ep->worker;
    ucp_rsc_index_t num_cm_cmpts = ucp_worker_num_cm_cmpts(worker);
    UCS_STRING_BUFFER_ONSTACK(cms_strb, 64);
    char addr_str[UCS_SOCKADDR_STRING_LEN];
    ucp_wireup_ep_t *cm_wireup_ep;
    ucp_rsc_index_t next_cm_idx;
    int i;

    next_cm_idx = ucp_cm_client_get_next_cm_idx(ep);
    if (next_cm_idx == UCP_NULL_RESOURCE) {
        for (i = 0; i < num_cm_cmpts; ++i) {
            ucs_string_buffer_appendf(&cms_strb, "%s,",
                                      ucp_context_cm_name(worker->context, i));
        }
        ucs_string_buffer_rtrim(&cms_strb, ",");

        cm_wireup_ep = ucp_ep_get_cm_wireup_ep(ep);
        ucs_assert_always(cm_wireup_ep != NULL);

        ucs_diag("client ep %p failed to connect to %s using %s cms",
                 ep,
                 ucs_sockaddr_str(
                         (struct sockaddr*)&cm_wireup_ep->cm_remote_sockaddr,
                         addr_str, sizeof(addr_str)),
                 ucs_string_buffer_cstr(&cms_strb));

        return 0;
    }

    ep->ext->cm_idx = next_cm_idx;
    ucs_callbackq_add_oneshot(&worker->uct->progress_q, ep,
                              ucp_cm_client_try_next_cm_progress, ep);
    ucp_worker_signal_internal(worker);
    return 1;
}

static ucp_rsc_index_t
ucp_cm_tl_bitmap_get_dev_idx(ucp_context_h context,
                             const ucp_tl_bitmap_t *tl_bitmap)
{
    ucp_rsc_index_t rsc_index = UCS_STATIC_BITMAP_FFS(*tl_bitmap);
    ucp_rsc_index_t dev_index;

    ucs_assert(!UCS_STATIC_BITMAP_IS_ZERO(*tl_bitmap));
    ucs_assert(rsc_index < context->num_tls);

    dev_index = context->tl_rscs[rsc_index].dev_index;

    /* check that all TL resources in the TL bitmap have the same dev_index */
    UCS_STATIC_BITMAP_FOR_EACH_BIT(rsc_index, tl_bitmap) {
        ucs_assert(dev_index == context->tl_rscs[rsc_index].dev_index);
    }

    return dev_index;
}

static ucs_status_t
ucp_cm_ep_client_initial_config_get(ucp_ep_h ucp_ep, unsigned ep_init_flags,
                                    const ucp_tl_bitmap_t *tl_bitmap,
                                    ucp_ep_config_key_t *key)
{
    ucp_worker_h worker           = ucp_ep->worker;
    ucp_context_h context         = worker->context;
    unsigned addr_pack_flags      = ucp_worker_common_address_pack_flags(worker) |
                                    UCP_ADDRESS_PACK_FLAG_DEVICE_ADDR |
                                    UCP_ADDRESS_PACK_FLAG_IFACE_ADDR;
    ucp_wireup_ep_t *wireup_ep    = ucp_ep_get_cm_wireup_ep(ucp_ep);
    unsigned wireup_ep_init_flags = wireup_ep->ep_init_flags | ep_init_flags;
    void *ucp_addr;
    size_t ucp_addr_size;
    ucp_unpacked_address_t unpacked_addr;
    ucp_address_entry_t *ae;
    unsigned addr_indices[UCP_MAX_RESOURCES];
    ucs_status_t status;

    ucs_debug("ep %p: init_flags 0x%x", ucp_ep, wireup_ep_init_flags);

    /* Construct local dummy address for lanes selection taking an assumption
     * that server has the transports which are the best from client's
     * perspective. */
    status = ucp_address_pack(worker, NULL, tl_bitmap, addr_pack_flags,
                              context->config.ext.worker_addr_version, NULL,
                              UINT_MAX, &ucp_addr_size, &ucp_addr);
    if (status != UCS_OK) {
        goto out;
    }

    status = ucp_address_unpack(worker, ucp_addr, addr_pack_flags,
                                &unpacked_addr);
    if (status != UCS_OK) {
        goto free_ucp_addr;
    }

    /* Update destination MD and RSC indices in the unpacked address list */
    ucp_unpacked_address_for_each(ae, &unpacked_addr) {
        ae->md_index                 = UCP_NULL_RESOURCE;
    }

    ucs_assert(unpacked_addr.address_count <= UCP_MAX_RESOURCES);
    ucp_ep_config_key_reset(key);
    ucp_ep_config_key_set_err_mode(key, wireup_ep_init_flags);
    ucp_ep_config_key_init_flags(key, wireup_ep_init_flags);
    status = ucp_wireup_select_lanes(ucp_ep, wireup_ep_init_flags, *tl_bitmap,
                                     &unpacked_addr, addr_indices, key, 0);

    ucs_free(unpacked_addr.address_list);
free_ucp_addr:
    ucs_free(ucp_addr);
out:
    return status;
}

size_t ucp_cm_sa_data_length(ucp_object_version_t sa_data_version)
{
    ucs_assert((sa_data_version == UCP_OBJECT_VERSION_V1) ||
               (sa_data_version == UCP_OBJECT_VERSION_V2));

    return (sa_data_version == UCP_OBJECT_VERSION_V1) ?
           sizeof(ucp_wireup_sockaddr_data_v1_t) :
           sizeof(ucp_wireup_sockaddr_data_base_t);
}

static size_t
ucp_cm_priv_data_length(size_t addr_size, ucp_object_version_t sa_data_version)
{
    return addr_size + ucp_cm_sa_data_length(sa_data_version);
}

unsigned ucp_cm_address_pack_flags(ucp_worker_h worker)
{
    return ucp_worker_common_address_pack_flags(worker) |
           UCP_ADDRESS_PACK_FLAGS_CM_DEFAULT |
           /* Pack TL resource indices into CM private data to use them inside
            * ucp_wireup_init_lanes() for checking intersection of endpoint
            * configurations */
           (worker->context->config.ext.cm_use_all_devices ?
            UCP_ADDRESS_PACK_FLAG_TL_RSC_IDX : 0);
}

static void*
ucp_cm_ep_sa_data_pack(ucp_ep_h ep, ucp_wireup_sockaddr_data_base_t *sa_data,
                       ucp_object_version_t sa_data_version)
{
    ucp_wireup_sockaddr_data_v1_t *sa_data_v1;

    if (sa_data_version == UCP_OBJECT_VERSION_V1) {
        sa_data->header       = ucp_ep_config(ep)->key.err_mode;
        sa_data_v1            = ucs_derived_of(sa_data,
                                               ucp_wireup_sockaddr_data_v1_t);
        sa_data_v1->addr_mode = UCP_WIREUP_SA_DATA_CM_ADDR;
        sa_data_v1->dev_index = 0;
        return sa_data_v1 + 1;
    } else {
        ucs_assertv_always(sa_data_version == UCP_OBJECT_VERSION_V2,
                           "sa_data version: %u", sa_data_version);
        sa_data->header = UCP_OBJECT_VERSION_V2 <<
                          UCP_SA_DATA_HEADER_VERSION_SHIFT;
        if (ucp_ep_config(ep)->key.err_mode == UCP_ERR_HANDLING_MODE_PEER) {
            sa_data->header |= UCP_SA_DATA_FLAG_ERR_MODE_PEER;
        }

        return sa_data + 1;
    }
}

static unsigned
ucp_wireup_cm_address_pack_flags(const ucp_worker_h worker,
                                 const ucp_wireup_ep_t *cm_wireup_ep,
                                 unsigned ep_init_flags)
{
    unsigned pack_flags = ucp_cm_address_pack_flags(worker);

    if (cm_wireup_ep->flags & UCP_WIREUP_EP_FLAG_SEND_CLIENT_ID) {
        pack_flags |= UCP_ADDRESS_PACK_FLAG_CLIENT_ID;
    }

    if (ep_init_flags & UCP_EP_INIT_CREATE_AM_LANE_ONLY) {
        pack_flags |= UCP_ADDRESS_PACK_FLAG_AM_ONLY;
    }

    return pack_flags;
}

static ucs_status_t
ucp_wireup_cm_priv_space_check(const ucp_worker_h worker,
                               ucp_rsc_index_t cm_idx, size_t ucp_addr_size,
                               ucp_object_version_t sa_data_version,
                               ucs_log_level_t log_level)
{
    size_t sa_data_length = ucp_cm_priv_data_length(ucp_addr_size,
                                                    sa_data_version);

    if (worker->cms[cm_idx].attr.max_conn_priv < sa_data_length) {
        ucs_log(log_level,
                "CM %s private data buffer is too small to pack UCP "
                "endpoint info, cm max_conn_priv %lu, service data version %u, "
                "size %lu, address length %lu",
                ucp_context_cm_name(worker->context, cm_idx),
                worker->cms[cm_idx].attr.max_conn_priv, sa_data_version,
                sa_data_length - ucp_addr_size, ucp_addr_size);
        return UCS_ERR_BUFFER_TOO_SMALL;
    }

    return UCS_OK;
}

static ucs_status_t
ucp_cm_ep_priv_data_pack(ucp_ep_h ep, const ucp_tl_bitmap_t *tl_bitmap,
                         ucs_log_level_t log_level,
                         ucp_object_version_t sa_data_version,
                         void **data_buf_p, size_t *data_buf_length_p,
                         unsigned pack_flags, unsigned max_num_paths)
{
    ucp_worker_h worker   = ep->worker;
    ucp_context_h context = worker->context;
    void *ucp_addr        = NULL;
    ucp_wireup_sockaddr_data_base_t *sa_data;
    ucs_status_t status;
    void *worker_addr_p;
    size_t ucp_addr_size, sa_data_length;

    ucs_assert((int)ucp_ep_config(ep)->key.err_mode <= UINT8_MAX);

    /* Don't pack the device address to reduce address size, it will be
     * delivered by uct_cm_listener_conn_request_callback_t in
     * uct_cm_remote_data_t */
    status = ucp_address_pack(worker, ep, tl_bitmap, pack_flags,
                              context->config.ext.worker_addr_version, NULL,
                              max_num_paths, &ucp_addr_size, &ucp_addr);
    if (status != UCS_OK) {
        goto err;
    }

    status = ucp_wireup_cm_priv_space_check(worker, ep->ext->cm_idx,
                                            ucp_addr_size, sa_data_version,
                                            log_level);
    if (status != UCS_OK) {
        goto err;
    }

    sa_data_length = ucp_cm_priv_data_length(ucp_addr_size, sa_data_version);
    sa_data        = ucs_malloc(sa_data_length, "client_priv_data");
    if (sa_data == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    sa_data->ep_id = ucp_ep_local_id(ep);
    worker_addr_p  = ucp_cm_ep_sa_data_pack(ep, sa_data, sa_data_version);

    memcpy(worker_addr_p, ucp_addr, ucp_addr_size);

    *data_buf_p        = sa_data;
    *data_buf_length_p = sa_data_length;

err:
    ucs_free(ucp_addr);
    return status;
}

static void
ucp_wireup_cm_ep_cleanup(ucp_ep_t *ucp_ep)
{
    ucp_lane_index_t lane_idx;

    for (lane_idx = 0; lane_idx < ucp_ep_num_lanes(ucp_ep); ++lane_idx) {
        if (lane_idx == ucp_ep_get_cm_lane(ucp_ep)) {
            continue;
        }

        /* During discarding, UCT EP's pending queues 
         * are expected to be empty */
        ucp_worker_discard_uct_ep(
                ucp_ep, ucp_ep_get_lane(ucp_ep, lane_idx),
                ucp_ep_get_rsc_index(ucp_ep, lane_idx), UCT_FLUSH_FLAG_CANCEL,
                (uct_pending_purge_callback_t)ucs_empty_function_do_assert_void,
                NULL, (ucp_send_nbx_callback_t)ucs_empty_function, NULL);
        ucp_ep_set_lane(ucp_ep, lane_idx, NULL);
    }
}

static ucs_status_t ucp_cm_ep_init_lanes(ucp_ep_h ep,
                                         ucp_tl_bitmap_t *tl_bitmap)
{
    ucp_worker_h worker = ep->worker;
    ucs_status_t status = UCS_ERR_NO_RESOURCE;
    ucp_lane_index_t lane_idx;
    ucp_rsc_index_t rsc_idx;
    uint8_t path_index;
    uct_ep_h uct_ep;

    UCS_STATIC_BITMAP_RESET_ALL(tl_bitmap);
    for (lane_idx = 0; lane_idx < ucp_ep_num_lanes(ep); ++lane_idx) {
        if (lane_idx == ucp_ep_get_cm_lane(ep)) {
            continue;
        }

        rsc_idx = ucp_ep_get_rsc_index(ep, lane_idx);
        if (rsc_idx == UCP_NULL_RESOURCE) {
            continue;
        }

        status = ucp_wireup_ep_create(ep, &uct_ep);
        if (status != UCS_OK) {
            goto out;
        }

        ucp_ep_set_lane(ep, lane_idx, uct_ep);

        UCS_STATIC_BITMAP_SET(tl_bitmap, rsc_idx);
        if (ucp_ep_is_lane_p2p(ep, lane_idx)) {
            path_index = ucp_ep_get_path_index(ep, lane_idx);
            status     = ucp_wireup_ep_connect(ucp_ep_get_lane(ep, lane_idx), 0,
                                               rsc_idx, path_index, 0, NULL);
            if (status != UCS_OK) {
                goto out;
            }

            ucp_worker_iface_progress_ep(ucp_worker_iface(worker, rsc_idx));
        } else {
            ucs_assert(ucp_worker_is_tl_2iface(worker, rsc_idx));
        }
    }

out:
    return status;
}

static unsigned ucp_cm_client_uct_connect_progress(void *arg)
{
    static const unsigned ep_init_flags_prio[] = {
        0, UCP_EP_INIT_KA_FROM_EXIST_LANES, UCP_EP_INIT_CREATE_AM_LANE_ONLY
    };
    ucp_ep_h ep                                = arg;
    ucp_worker_h worker                        = ep->worker;
    ucp_context_h context                      = worker->context;
    ucp_wireup_ep_t *cm_wireup_ep              =
            ucp_ep_get_cm_wireup_ep(ep);
    unsigned ep_init_flags, ep_pack_flags, i;
    ucp_tl_bitmap_t tl_bitmap;
    uct_ep_connect_params_t params;
    void *priv_data;
    size_t priv_data_length;
    ucp_ep_config_key_t key;
    ucs_queue_head_t tmp_pending_queue;
    ucs_status_t status;
    ucs_log_level_t fallback_log_level;
    ucp_worker_cfg_index_t cfg_index;

    UCS_ASYNC_BLOCK(&worker->async);

    /* To silence Coverity warning */
    UCS_STATIC_ASSERT(ucs_static_array_size(ep_init_flags_prio) > 0);

    for (i = 0; i < ucs_static_array_size(ep_init_flags_prio); ++i) {
        ep_init_flags = ep_init_flags_prio[i];
        status        = ucp_cm_ep_client_initial_config_get(
                ep, ep_init_flags, &cm_wireup_ep->cm_resolve_tl_bitmap, &key);
        if (status != UCS_OK) {
            goto try_fallback;
        }

        ucp_ep_get_tl_bitmap(&key, &tl_bitmap);
        ep_pack_flags = ucp_wireup_cm_address_pack_flags(worker, cm_wireup_ep,
                                                         ep_init_flags);
        status        = ucp_address_length(worker, &key, &tl_bitmap, ep_pack_flags,
                                           context->config.ext.worker_addr_version,
                                           &priv_data_length);
        if (status != UCS_OK) {
            goto err;
        }

        if (i < (ucs_static_array_size(ep_init_flags_prio) - 1)) {
            /* Can use another variant of endpoint's initialization flags for
             * fallback */
            fallback_log_level = UCS_LOG_LEVEL_DEBUG;
        } else if (ucp_cm_client_get_next_cm_idx(ep) != UCP_NULL_RESOURCE) {
            /* Can use another CM for fallback */
            fallback_log_level = UCS_LOG_LEVEL_DIAG;
        } else {
            /* No options for fallback */
            fallback_log_level = UCS_LOG_LEVEL_ERROR;
        }

        status = ucp_wireup_cm_priv_space_check(
                worker, ep->ext->cm_idx, priv_data_length,
                context->config.ext.sa_client_min_hdr_version,
                fallback_log_level);
        if (status == UCS_ERR_BUFFER_TOO_SMALL) {
            continue;
        } else if (status != UCS_OK) {
            goto err;
        }

        ucp_wireup_eps_pending_extract(ep, &tmp_pending_queue);
        ucp_wireup_cm_ep_cleanup(ep);
        ucp_ep_realloc_lanes(ep, key.num_lanes);

        status = ucp_worker_get_ep_config(worker, &key, ep_init_flags,
                                          &cfg_index);
        if (status != UCS_OK) {
            goto err;
        }

        ucp_ep_set_cfg_index(ep, cfg_index);
        ep->am_lane = key.am_lane;

        status = ucp_cm_ep_init_lanes(ep, &tl_bitmap);
        if (status != UCS_OK) {
            goto err;
        }

        status = ucp_cm_ep_priv_data_pack(
                ep, &tl_bitmap, fallback_log_level,
                context->config.ext.sa_client_min_hdr_version, &priv_data,
                &priv_data_length, ep_pack_flags, UINT_MAX);
        if (status == UCS_OK) {
            break;
        } else if (status != UCS_ERR_BUFFER_TOO_SMALL) {
            goto err;
        }
    }

    if (status == UCS_ERR_BUFFER_TOO_SMALL) {
        ucs_assert(i == ucs_static_array_size(ep_init_flags_prio));
        goto try_fallback;
    }

    params.field_mask          = UCT_EP_CONNECT_PARAM_FIELD_PRIVATE_DATA |
                                 UCT_EP_CONNECT_PARAM_FIELD_PRIVATE_DATA_LENGTH;
    params.private_data        = priv_data;
    params.private_data_length = priv_data_length;
    status                     = uct_ep_connect(ucp_ep_get_cm_uct_ep(ep),
                                                &params);
    ucs_free(priv_data);

    ucp_wireup_replay_pending_requests(ep, &tmp_pending_queue);
    if (status != UCS_OK) {
        goto err;
    }

    ucp_ep_update_flags(ep, UCP_EP_FLAG_LOCAL_CONNECTED, 0);
    goto out;

try_fallback:
    if (ucp_cm_client_try_fallback_cms(ep)) {
        /* Can fallback to the next CM to retry getting CM initial config to
         * fit to CM private data */
        goto out;
    }

err:
    ucp_ep_set_failed(ep, ucp_ep_get_cm_lane(ep), status);
out:
    UCS_ASYNC_UNBLOCK(&worker->async);
    return 1;
}

static ucs_status_t
ucp_cm_client_resolve_cb(void *user_data, const uct_cm_ep_resolve_args_t *args)
{
    ucp_ep_h ep         = user_data;
    ucp_worker_h worker = ep->worker;
    ucs_status_t status = UCS_PARAM_VALUE(UCT_CM_EP_RESOLVE_ARGS_FIELD, args,
                                          status, STATUS, UCS_OK);
    ucp_wireup_ep_t *cm_wireup_ep;
    char addr_str[UCS_SOCKADDR_STRING_LEN];

    UCP_EP_CM_CALLBACK_ENTER(ep, ucp_ep_get_cm_uct_ep(ep),
                             {
                                 ucs_assert(ep->flags & UCP_EP_FLAG_CLOSED);
                                 status = UCS_ERR_CANCELED;
                                 goto out;
                             });

    if (status != UCS_OK) {
        goto try_fallback;
    }

    ucs_assert_always(args->field_mask & UCT_CM_EP_RESOLVE_ARGS_FIELD_DEV_NAME);

    cm_wireup_ep = ucp_ep_get_cm_wireup_ep(ep);
    ucs_assert(cm_wireup_ep != NULL);
    ucp_context_dev_tl_bitmap(worker->context, args->dev_name,
                              &cm_wireup_ep->cm_resolve_tl_bitmap);

    if (UCS_STATIC_BITMAP_IS_ZERO(cm_wireup_ep->cm_resolve_tl_bitmap)) {
        ucs_diag("client ep %p connect to %s failed: device %s is not enabled, "
                 "enable it in UCX_NET_DEVICES or use corresponding ip address",
                 ep,
                 ucs_sockaddr_str(
                        (struct sockaddr*)&cm_wireup_ep->cm_remote_sockaddr,
                         addr_str, sizeof(addr_str)),
                 args->dev_name);
        status = UCS_ERR_UNREACHABLE;
        goto try_fallback;
    }

    ucs_debug("client created ep %p on device %s, "
              "tl_bitmap " UCT_TL_BITMAP_FMT " on cm %s",
              ep, args->dev_name,
              UCT_TL_BITMAP_ARG(&cm_wireup_ep->cm_resolve_tl_bitmap),
              ucp_context_cm_name(worker->context, ep->ext->cm_idx));

    ucs_callbackq_add_oneshot(&worker->uct->progress_q, ep,
                              ucp_cm_client_uct_connect_progress, ep);
    ucp_worker_signal_internal(worker);
    goto out;

try_fallback:
    if (!ucp_cm_client_try_fallback_cms(ep)) {
        ucp_ep_set_failed_schedule(ep, ucp_ep_get_cm_lane(ep), status);
    }
out:
    return status;
}

static void
ucp_cm_client_connect_prog_arg_free(ucp_cm_client_connect_progress_arg_t *arg)
{
    ucs_free(arg->sa_data);
    ucs_free(arg->dev_addr);
    ucs_free(arg);
}

/*
 * The main thread progress part of connection establishment on client side
 */
static unsigned ucp_cm_client_connect_progress(void *arg)
{
    ucp_cm_client_connect_progress_arg_t *progress_arg = arg;
    ucp_ep_h ucp_ep                                    = progress_arg->ucp_ep;
    ucp_worker_h worker                                = ucp_ep->worker;
    ucp_context_h context                              = worker->context;
    uct_ep_h uct_cm_ep                                 = ucp_ep_get_cm_uct_ep(ucp_ep);
    unsigned pack_flags                                = ucp_cm_address_pack_flags(worker);
    const void *ucp_address;
    ucp_wireup_ep_t *wireup_ep;
    ucp_unpacked_address_t addr;
    ucp_tl_bitmap_t tl_bitmap;
    ucp_rsc_index_t dev_index;
    unsigned addr_idx;
    unsigned addr_indices[UCP_MAX_RESOURCES];
    ucs_status_t status;
    uint8_t sa_data_ver;

    UCS_ASYNC_BLOCK(&worker->async);

    ucs_debug("ep %p flags 0x%x cfg_index %d: client connect progress", ucp_ep,
              ucp_ep->flags, ucp_ep->cfg_index);
    ucs_log_indent(1);

    ucs_assert(ucp_ep->flags & UCP_EP_FLAG_LOCAL_CONNECTED);

    wireup_ep = ucp_ep_get_cm_wireup_ep(ucp_ep);
    ucs_assert(wireup_ep != NULL);
    ucs_assert(wireup_ep->ep_init_flags & UCP_EP_INIT_CM_WIREUP_CLIENT);

    sa_data_ver = progress_arg->sa_data->header >>
                  UCP_SA_DATA_HEADER_VERSION_SHIFT;
    ucp_address = UCS_PTR_BYTE_OFFSET(progress_arg->sa_data,
                                      ucp_cm_sa_data_length(sa_data_ver));
    status      = ucp_address_unpack(worker, ucp_address, pack_flags, &addr);
    if (status != UCS_OK) {
        goto out;
    }

    if (addr.address_count == 0) {
        status = UCS_ERR_UNREACHABLE;
        goto out_free_addr;
    }

    for (addr_idx = 0; addr_idx < addr.address_count; ++addr_idx) {
        addr.address_list[addr_idx].dev_addr  = progress_arg->dev_addr;
        addr.address_list[addr_idx].dev_index = 0; /* CM addr contains only 1
                                                      device */
    }

    if (ucp_address_is_am_only(ucp_address)) {
        wireup_ep->ep_init_flags |= UCP_EP_INIT_CREATE_AM_LANE_ONLY;
    }

    ucs_assert(addr.address_count <= UCP_MAX_RESOURCES);
    ucp_ep_update_remote_id(ucp_ep, progress_arg->sa_data->ep_id);

    ucp_ep_get_tl_bitmap(&ucp_ep_config(ucp_ep)->key, &tl_bitmap);
    dev_index = ucp_cm_tl_bitmap_get_dev_idx(worker->context, &tl_bitmap);

    ucp_context_dev_idx_tl_bitmap(context, dev_index, &tl_bitmap);
    status    = ucp_wireup_init_lanes(ucp_ep, wireup_ep->ep_init_flags,
                                      &tl_bitmap, &addr, addr_indices);
    if (status != UCS_OK) {
        ucs_debug("ep %p: failed to initialize lanes: %s", ucp_ep,
                  ucs_status_string(status));
        goto out_free_addr;
    }

    status = ucp_wireup_connect_local(ucp_ep, &addr, NULL);
    if (status != UCS_OK) {
        ucs_debug("ep %p: failed to connect lanes: %s", ucp_ep,
                  ucs_status_string(status));
        goto out_free_addr;
    }

    status = uct_cm_client_ep_conn_notify(uct_cm_ep);
    if (status != UCS_OK) {
        ucs_debug("ep %p: failed to send notify: %s", ucp_ep,
                  ucs_status_string(status));
        /* connection can't be established by UCT, no need to disconnect */
        ucp_ep_update_flags(ucp_ep, 0, UCP_EP_FLAG_LOCAL_CONNECTED);
        goto out_free_addr;
    }

    if (context->config.ext.cm_use_all_devices) {
        ucp_ep->flags |= UCP_EP_FLAG_CONNECT_WAIT_PRE_REQ;
        ucp_wireup_update_flags(ucp_ep, UCP_WIREUP_EP_FLAG_REMOTE_CONNECTED);
    } else {
        ucp_wireup_remote_connected(ucp_ep);
    }

out_free_addr:
    ucs_free(addr.address_list);
out:
    if (status != UCS_OK) {
        ucp_ep_set_failed(ucp_ep, ucp_ep_get_cm_lane(ucp_ep), status);
    }

    ucs_log_indent(-1);
    UCS_ASYNC_UNBLOCK(&worker->async);
    ucp_cm_client_connect_prog_arg_free(progress_arg);
    return 1;
}

static ucs_status_t
ucp_cm_remote_data_check(const uct_cm_remote_data_t *remote_data)
{
    if (ucs_test_all_flags(remote_data->field_mask,
                           UCT_CM_REMOTE_DATA_FIELD_DEV_ADDR        |
                           UCT_CM_REMOTE_DATA_FIELD_DEV_ADDR_LENGTH |
                           UCT_CM_REMOTE_DATA_FIELD_CONN_PRIV_DATA  |
                           UCT_CM_REMOTE_DATA_FIELD_CONN_PRIV_DATA_LENGTH)) {
        return UCS_OK;
    }

    ucs_error("incompatible client server connection establishment protocol "
              "(field_mask %"PRIu64")", remote_data->field_mask);
    return UCS_ERR_UNSUPPORTED;
}

/*
 * Async callback on a client side which notifies that server is connected.
 */
static void ucp_cm_client_connect_cb(uct_ep_h uct_cm_ep, void *arg,
                                     const uct_cm_ep_client_connect_args_t
                                     *connect_args)
{
    ucp_ep_h ucp_ep     = (ucp_ep_h)arg;
    ucp_worker_h worker = ucp_ep->worker;
    ucp_cm_client_connect_progress_arg_t *progress_arg;
    const uct_cm_remote_data_t *remote_data;
    ucs_status_t status;

    ucs_assert_always(ucs_test_all_flags(connect_args->field_mask,
                                         (UCT_CM_EP_CLIENT_CONNECT_ARGS_FIELD_REMOTE_DATA |
                                          UCT_CM_EP_CLIENT_CONNECT_ARGS_FIELD_STATUS)));
    remote_data = connect_args->remote_data;
    status      = connect_args->status;
    ucp_ep_update_flags(ucp_ep, UCP_EP_FLAG_CLIENT_CONNECT_CB, 0);

    ucs_debug("ep %p flags 0x%x cfg_index %d: client connected status %s",
              ucp_ep, ucp_ep->flags, ucp_ep->cfg_index,
              ucs_status_string(status));

    UCP_EP_CM_CALLBACK_ENTER(ucp_ep, uct_cm_ep, return);

    if (((status == UCS_ERR_NOT_CONNECTED) || (status == UCS_ERR_UNREACHABLE) ||
         (status == UCS_ERR_CONNECTION_RESET)) &&
        /* try connecting through another cm (next one in the priority list) */
        ucp_cm_client_try_fallback_cms(ucp_ep)) {
        /* connection can't be established by UCT, no need to disconnect */
        ucp_ep_update_flags(ucp_ep, 0, UCP_EP_FLAG_LOCAL_CONNECTED);
        /* cms fallback has started */
        return;
    } else if (status != UCS_OK) {
        /* connection can't be established by UCT, no need to disconnect */
        ucp_ep_update_flags(ucp_ep, 0, UCP_EP_FLAG_LOCAL_CONNECTED);
        ucs_debug("failed status on client connect callback: %s "
                  "(sockaddr_cm=%s, cms_used_idx=%d)",
                  ucs_status_string(status),
                  ucp_context_cm_name(worker->context, ucp_ep->ext->cm_idx),
                  ucp_ep->ext->cm_idx);
        goto err_out;
    }

    status = ucp_cm_remote_data_check(remote_data);
    if (status != UCS_OK) {
        goto err_out;
    }

    progress_arg = ucs_malloc(sizeof(*progress_arg),
                              "ucp_cm_client_connect_progress_arg_t");
    if (progress_arg == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_out;
    }

    progress_arg->sa_data = ucs_malloc(remote_data->conn_priv_data_length,
                                       "sa data");
    if (progress_arg->sa_data == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free_arg;
    }

    progress_arg->dev_addr = ucs_malloc(remote_data->dev_addr_length,
                                        "device address");
    if (progress_arg->dev_addr == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free_sa_data;
    }

    progress_arg->ucp_ep = ucp_ep;
    memcpy(progress_arg->dev_addr, remote_data->dev_addr,
           remote_data->dev_addr_length);
    memcpy(progress_arg->sa_data, remote_data->conn_priv_data,
           remote_data->conn_priv_data_length);

    ucs_callbackq_add_oneshot(&worker->uct->progress_q, ucp_ep,
                              ucp_cm_client_connect_progress, progress_arg);
    ucp_worker_signal_internal(ucp_ep->worker);
    return;

err_free_sa_data:
    ucs_free(progress_arg->sa_data);
err_free_arg:
    ucs_free(progress_arg);
err_out:
    ucp_ep_set_failed_schedule(ucp_ep, ucp_ep_get_cm_lane(ucp_ep), status);
}

static void ucp_ep_cm_remote_disconnect_progress(ucp_ep_h ucp_ep)
{
    ucs_status_t status = UCS_ERR_CONNECTION_RESET;

    ucs_trace("ep %p: flags 0x%x cm_remote_disconnect_progress", ucp_ep,
              ucp_ep->flags);

    ucs_assert(ucp_ep_get_cm_uct_ep(ucp_ep) != NULL);

    ucs_assert(ucp_ep->flags & UCP_EP_FLAG_LOCAL_CONNECTED);
    if ((ucp_ep->flags & UCP_EP_FLAG_CLOSED) &&
        (ucp_ep->ext->close_req != NULL)) {
        ucp_request_complete_send(ucp_ep->ext->close_req, UCS_OK);
        return;
    }

    if (!(ucp_ep->flags & UCP_EP_FLAG_REMOTE_CONNECTED)) {
        /* CM disconnect happens during WIREUP MSGs exchange phase, when EP is
         * locally connected to the peer, so UCP EP should not wait for flush
         * completion even if it was started from close EP procedure, because
         * it won't be never completed due to unreachability of the peer */
        goto set_ep_failed;
    }

    if (ucp_ep->flags & UCP_EP_FLAG_CLOSED) {
        /* the ep is remote connected (checked above) and closed by API but
         * close req is not valid yet (checked above), it will be set later
         * from scheduled @ref ucp_ep_close_flushed_callback */
        ucs_debug("ep %p: ep is remote connected and closed, but request is"
                  " not set, waiting for the flush callback", ucp_ep);
        return;
    }

set_ep_failed:
    ucp_ep_set_failed(ucp_ep, ucp_ep_get_cm_lane(ucp_ep), status);
}

static unsigned ucp_ep_cm_disconnect_progress(void *arg)
{
    ucp_ep_h ucp_ep            = arg;
    uct_ep_h uct_cm_ep         = ucp_ep_get_cm_uct_ep(ucp_ep);
    ucs_async_context_t *async = &ucp_ep->worker->async;
    ucp_request_t *close_req;

    UCS_ASYNC_BLOCK(async);

    ucs_trace("ep %p: got remote disconnect, cm_ep %p, flags 0x%x", ucp_ep,
              uct_cm_ep, ucp_ep->flags);
    ucs_assert(ucp_ep_get_cm_uct_ep(ucp_ep) == uct_cm_ep);

    if (ucp_ep->flags & UCP_EP_FLAG_FAILED) {
        /* - ignore close event on failed ep, since all lanes are destroyed in
             generic err flow
           - if close req is valid this is ucp_ep_close_nb request and it will
             be completed as the ep is destroyed, i.e. flushed and disconnected
             with any status */
        if (ucp_ep->ext->close_req != NULL) {
            ucs_assert(ucp_ep->flags & UCP_EP_FLAG_CLOSED);
        }
    } else if (ucp_ep->flags & UCP_EP_FLAG_LOCAL_CONNECTED) {
        ucp_ep_cm_remote_disconnect_progress(ucp_ep);
    } else if (ucp_ep->ext->close_req != NULL) {
        /* if the EP is not local connected, the EP has been closed and flushed,
           CM lane is disconnected, complete close request and destroy EP */
        ucs_assert(ucp_ep->flags & UCP_EP_FLAG_CLOSED);
        ucp_ep_update_flags(ucp_ep, 0, UCP_EP_FLAG_REMOTE_CONNECTED);
        close_req = ucp_ep->ext->close_req;
        ucp_ep_local_disconnect_progress(close_req);
        /* don't touch UCP EP after local disconnect, since it is not valid
         * anymore */
        goto out;
    } else if (ucp_ep->flags & UCP_EP_FLAG_CLOSED) {
        /* if an EP was closed and not local connected anymore (i.e.
         * ucp_ep_cm_disconnect_cm_lane() was called from ucp_ep_close_nbx()),
         * not failed and no CLOSE request is set, it means that an EP was
         * disconnected from a peer */
        ucs_assert(ucp_ep->flags & UCP_EP_FLAG_DISCONNECTED_CM_LANE);
        ucs_assert(!(ucp_ep->flags & UCP_EP_FLAG_ERR_HANDLER_INVOKED));
    } else {
        ucs_warn("ep %p: unexpected state on disconnect, flags: 0x%u",
                 ucp_ep, ucp_ep->flags);
    }

    /* don't remove the flag at the beginning of the function, some functions
     * may rely on that flag (e.g. ucp_ep_cm_remote_disconnect_progress()) */
    ucp_ep_update_flags(ucp_ep, 0, UCP_EP_FLAG_REMOTE_CONNECTED);

out:
    UCS_ASYNC_UNBLOCK(async);
    return 1;
}

static void ucp_cm_disconnect_cb(uct_ep_h uct_cm_ep, void *arg)
{
    ucp_ep_h ucp_ep     = arg;
    ucp_worker_h worker = ucp_ep->worker;
    uct_ep_h uct_ep;

    ucp_ep_update_flags(ucp_ep, UCP_EP_FLAG_DISCONNECT_CB_CALLED, 0);
    ucs_trace("ep %p flags 0x%x: remote disconnect callback invoked", ucp_ep,
              ucp_ep->flags);

    UCP_EP_CM_CALLBACK_ENTER(ucp_ep, uct_cm_ep, return);

    uct_ep = ucp_ep_get_cm_uct_ep(ucp_ep);
    ucs_assertv_always(uct_cm_ep == uct_ep,
                       "%p: uct_cm_ep=%p vs found_uct_ep=%p",
                       ucp_ep, uct_cm_ep, uct_ep);

    ucs_callbackq_add_oneshot(&worker->uct->progress_q, ucp_ep,
                              ucp_ep_cm_disconnect_progress, ucp_ep);
    ucp_worker_signal_internal(worker);
}

ucs_status_t ucp_ep_client_cm_create_uct_ep(ucp_ep_h ucp_ep)
{
    ucp_wireup_ep_t *wireup_ep = ucp_ep_get_cm_wireup_ep(ucp_ep);
    ucp_rsc_index_t cm_idx     = ucp_ep->ext->cm_idx;
    ucp_worker_h worker        = ucp_ep->worker;
    uct_ep_params_t cm_lane_params;
    ucs_sock_addr_t remote_addr, local_addr;
    size_t sockaddr_size;
    ucs_status_t status;
    uct_ep_h cm_ep;

    cm_lane_params.field_mask = UCT_EP_PARAM_FIELD_CM                         |
                                UCT_EP_PARAM_FIELD_USER_DATA                  |
                                UCT_EP_PARAM_FIELD_SOCKADDR                   |
                                UCT_EP_PARAM_FIELD_SOCKADDR_CB_FLAGS          |
                                UCT_EP_PARAM_FIELD_CM_RESOLVE_CB              |
                                UCT_EP_PARAM_FIELD_SOCKADDR_CONNECT_CB_CLIENT |
                                UCT_EP_PARAM_FIELD_SOCKADDR_DISCONNECT_CB;

    status = ucs_sockaddr_sizeof((struct sockaddr *)&wireup_ep->cm_remote_sockaddr,
                                 &sockaddr_size);
    if (status != UCS_OK) {
        return status;
    }

    remote_addr.addrlen               = sockaddr_size;
    remote_addr.addr                  = (struct sockaddr *)&wireup_ep->cm_remote_sockaddr;

    cm_lane_params.sockaddr           = &remote_addr;
    cm_lane_params.user_data          = ucp_ep;
    cm_lane_params.sockaddr_cb_flags  = UCT_CB_FLAG_ASYNC;
    cm_lane_params.cm_resolve_cb      = ucp_cm_client_resolve_cb;
    cm_lane_params.sockaddr_cb_client = ucp_cm_client_connect_cb;
    cm_lane_params.disconnect_cb      = ucp_cm_disconnect_cb;
    cm_lane_params.cm                 = worker->cms[cm_idx].cm;

    if (wireup_ep->cm_local_sockaddr.ss_family != 0) {
        /* user specified local address */
        status = ucs_sockaddr_sizeof((const struct sockaddr*)&wireup_ep->cm_local_sockaddr,
                                     &sockaddr_size);
        if (status != UCS_OK) {
            return status;
        }

        local_addr.addrlen            = sockaddr_size;
        local_addr.addr               = (struct sockaddr*)&wireup_ep->cm_local_sockaddr;
        cm_lane_params.field_mask    |= UCT_EP_PARAM_FIELD_LOCAL_SOCKADDR;
        cm_lane_params.local_sockaddr = &local_addr;
    }

    status = uct_ep_create(&cm_lane_params, &cm_ep);
    if (status != UCS_OK) {
        /* coverity[leaked_storage] */
        return status;
    }

    ucp_wireup_ep_set_next_ep(&wireup_ep->super.super, cm_ep, UCP_NULL_RESOURCE);
    ucs_trace("created cm_ep %p, wireup_ep %p, uct_ep %p, wireup_ep_from_uct_ep %p",
              cm_ep, wireup_ep, &wireup_ep->super.super, ucp_wireup_ep(&wireup_ep->super.super));
    return status;
}

ucs_status_t ucp_ep_client_cm_connect_start(ucp_ep_h ucp_ep,
                                            const ucp_ep_params_t *params)
{
    ucp_worker_h worker        = ucp_ep->worker;
    ucp_wireup_ep_t *wireup_ep = ucp_ep_get_cm_wireup_ep(ucp_ep);
    ucs_status_t status;

    ucs_assert(ucp_ep->ext->cm_idx == UCP_NULL_RESOURCE);

    ucp_ep->ext->cm_idx      = 0;
    wireup_ep->ep_init_flags = ucp_ep_init_flags(worker, params);

    /* save the address from the ep_params on the wireup_ep */
    status = ucs_sockaddr_copy((struct sockaddr*)&wireup_ep->cm_remote_sockaddr,
                               params->sockaddr.addr);
    if (status != UCS_OK) {
        return status;
    }

    if (params->field_mask & UCP_EP_PARAM_FIELD_LOCAL_SOCK_ADDR) {
        /* save local address from the ep_params on the wireup_ep */
        status = ucs_sockaddr_copy((struct sockaddr*)&wireup_ep->cm_local_sockaddr,
                                   params->local_sockaddr.addr);
        if (status != UCS_OK) {
            return status;
        }
    } else {
        memset(&wireup_ep->cm_local_sockaddr, 0, sizeof(wireup_ep->cm_local_sockaddr));
    }

    status = ucp_ep_client_cm_create_uct_ep(ucp_ep);
    if ((status != UCS_OK) && !ucp_cm_client_try_fallback_cms(ucp_ep)) {
        ucp_ep_set_failed_schedule(ucp_ep, ucp_ep_get_cm_lane(ucp_ep), status);
    }

    return UCS_OK;
}

static unsigned ucp_cm_server_conn_request_progress(void *arg)
{
    ucp_conn_request_h               conn_request = arg;
    ucp_listener_h                   listener     = conn_request->listener;
    ucp_worker_h                     worker       = listener->worker;
    ucp_ep_h                         ep;

    ucs_trace_func("listener %p, connect request %p", listener, conn_request);

    if (listener->conn_cb) {
        listener->conn_cb(conn_request, listener->arg);
        return 1;
    }

    ucs_assert(listener->accept_cb != NULL);
    UCS_ASYNC_BLOCK(&worker->async);
    ucp_ep_create_server_accept(worker, conn_request, &ep);
    UCS_ASYNC_UNBLOCK(&worker->async);
    return 1;
}

static ucp_rsc_index_t ucp_listener_get_cm_index(uct_listener_h listener,
                                                 ucp_worker_h worker)
{
    const ucp_rsc_index_t num_cms = ucp_worker_num_cm_cmpts(worker);
    ucp_rsc_index_t i;

    for (i = 0; i < num_cms; i++) {
        if (worker->cms[i].cm == listener->cm) {
            return i;
        }
    }

    return UCP_NULL_RESOURCE;
}

int ucp_cm_server_conn_request_progress_cb_pred(const ucs_callbackq_elem_t *elem,
                                                void *arg)
{
    ucp_listener_h listener = arg;
    ucp_conn_request_h conn_request;

    if (elem->cb != ucp_cm_server_conn_request_progress) {
        return 0;
    }

    conn_request = elem->arg;
    if (conn_request->listener != listener) {
        return 0;
    }

    ucp_listener_reject(listener, conn_request);
    return 1;
}

void ucp_cm_server_conn_request_cb(uct_listener_h listener, void *arg,
                                   const uct_cm_listener_conn_request_args_t
                                   *conn_req_args)
{
    ucp_listener_h ucp_listener = arg;
    ucp_worker_h worker         = ucp_listener->worker;
    ucp_conn_request_h ucp_conn_request;
    uct_conn_request_h conn_request;
    const uct_cm_remote_data_t *remote_data;
    ucp_rsc_index_t cm_idx;
    ucs_status_t status;

    ucs_assert_always(ucs_test_all_flags(conn_req_args->field_mask,
                                         (UCT_CM_LISTENER_CONN_REQUEST_ARGS_FIELD_CONN_REQUEST |
                                          UCT_CM_LISTENER_CONN_REQUEST_ARGS_FIELD_REMOTE_DATA  |
                                          UCT_CM_LISTENER_CONN_REQUEST_ARGS_FIELD_DEV_NAME     |
                                          UCT_CM_LISTENER_CONN_REQUEST_ARGS_FIELD_CLIENT_ADDR)));

    ucp_listener->conn_reqs++;
    conn_request = conn_req_args->conn_request;
    remote_data  = conn_req_args->remote_data;

    status = ucp_cm_remote_data_check(remote_data);
    if (status != UCS_OK) {
        goto err_reject;
    }

    cm_idx = ucp_listener_get_cm_index(listener, worker);
    if (cm_idx == UCP_NULL_RESOURCE) {
        ucs_error("failed to find listener's cm (%p) in local cms array",
                  listener->cm);
        goto err_reject;
    }

    ucs_debug("server received a connection request on the %s "
              "sockaddr transport (worker=%p cm=%p worker_cms_index=%d)",
              ucp_context_cm_name(worker->context, cm_idx),
              worker, listener->cm, cm_idx);

    ucp_conn_request = ucs_malloc(sizeof(ucp_conn_request_t) +
                                  remote_data->conn_priv_data_length,
                                  "ucp_conn_request_h");
    if (ucp_conn_request == NULL) {
        ucs_error("failed to allocate connect request, rejecting connection "
                  "request %p on TL listener %p",
                  conn_request, listener);
        goto err_reject;
    }

    ucp_conn_request->remote_dev_addr = ucs_malloc(remote_data->dev_addr_length,
                                                   "remote device address");
    if (ucp_conn_request->remote_dev_addr == NULL) {
        ucs_error("failed to allocate device address, rejecting connection "
                  "request %p on TL listener %p",
                  conn_request, listener);
        goto err_free_ucp_conn_request;
    }

    ucp_conn_request->listener     = ucp_listener;
    ucp_conn_request->uct_listener = listener;
    ucp_conn_request->uct_req      = conn_request;
    ucp_conn_request->cm_idx       = cm_idx;
    ucp_conn_request->ep           = NULL;

    status = ucs_sockaddr_copy((struct sockaddr *)&ucp_conn_request->client_address,
                               conn_req_args->client_address.addr);
    if (status != UCS_OK) {
        goto err_free_remote_dev_addr;
    }

    ucs_strncpy_safe(ucp_conn_request->dev_name, conn_req_args->dev_name,
                     UCT_DEVICE_NAME_MAX);
    memcpy(ucp_conn_request->remote_dev_addr, remote_data->dev_addr,
           remote_data->dev_addr_length);
    memcpy(ucp_conn_request + 1, remote_data->conn_priv_data,
           remote_data->conn_priv_data_length);

    ucs_callbackq_add_oneshot(&worker->uct->progress_q, ucp_listener,
                              ucp_cm_server_conn_request_progress,
                              ucp_conn_request);

    /* If the worker supports the UCP_FEATURE_WAKEUP feature, signal the user so
     * that he can wake-up on this event */
    ucp_worker_signal_internal(worker);
    return;

err_free_remote_dev_addr:
    ucs_free(ucp_conn_request->remote_dev_addr);
err_free_ucp_conn_request:
    ucs_free(ucp_conn_request);
err_reject:
    ucp_listener->conn_reqs--;
    status = uct_listener_reject(listener, conn_request);
    if (status != UCS_OK) {
        /* coverity[pass_freed_arg] */
        ucs_warn("failed to reject connect request %p on listener %p",
                 conn_request, listener);
    }
}

ucs_status_t
ucp_ep_cm_server_create_connected(ucp_worker_h worker, unsigned ep_init_flags,
                                  const ucp_unpacked_address_t *remote_addr,
                                  ucp_conn_request_h conn_request,
                                  ucp_ep_h *ep_p)
{
    ucp_listener_h listener = conn_request->listener;
    ucp_tl_bitmap_t tl_bitmap;
    ucp_ep_h ep;
    ucs_status_t status;
    char client_addr_str[UCS_SOCKADDR_STRING_LEN];
    ucp_wireup_sockaddr_data_base_t *sa_data;
    ucp_object_version_t sa_data_version;
    unsigned addr_indices[UCP_MAX_LANES];

    ep_init_flags |= UCP_EP_INIT_CM_WIREUP_SERVER | UCP_EP_INIT_CM_PHASE;

    ucp_context_dev_tl_bitmap(worker->context, conn_request->dev_name,
                              &tl_bitmap);
    if (UCS_STATIC_BITMAP_IS_ZERO(tl_bitmap)) {
        ucs_error("listener %p: got connection request from %s on a device %s "
                  "which was not present during UCP initialization",
                  conn_request->listener,
                  ucs_sockaddr_str((struct sockaddr*)&conn_request->client_address,
                                   client_addr_str, sizeof(client_addr_str)),
                  conn_request->dev_name);
        uct_listener_reject(conn_request->uct_listener, conn_request->uct_req);
        status = UCS_ERR_UNREACHABLE;
        goto out_free_request;
    }

    /* Create and connect TL part */
    status = ucp_ep_create_to_worker_addr(worker, &tl_bitmap, remote_addr,
                                          ep_init_flags,
                                          "conn_request on uct_listener",
                                          addr_indices, &ep);
    if (status != UCS_OK) {
        ucs_warn("failed to create server ep and connect to worker address on "
                 "device %s, tl_bitmap " UCT_TL_BITMAP_FMT ", status %s",
                 conn_request->dev_name, UCT_TL_BITMAP_ARG(&tl_bitmap),
                 ucs_status_string(status));
        uct_listener_reject(conn_request->uct_listener, conn_request->uct_req);
        goto out_free_request;
    }

    status = ucp_wireup_connect_local(ep, remote_addr, NULL);
    if (status != UCS_OK) {
        ucs_warn("server ep %p failed to connect to remote address on "
                 "device %s, tl_bitmap " UCT_TL_BITMAP_FMT ", status %s",
                 ep, conn_request->dev_name, tl_bitmap.bits[0],
                 tl_bitmap.bits[1], ucs_status_string(status));
        uct_listener_reject(conn_request->uct_listener, conn_request->uct_req);
        goto err_destroy_ep;
    }

    sa_data         = (ucp_wireup_sockaddr_data_base_t*)(conn_request + 1);
    sa_data_version = sa_data->header >> UCP_SA_DATA_HEADER_VERSION_SHIFT;
    status          = ucp_ep_cm_connect_server_lane(
                          ep, conn_request->uct_listener, conn_request->uct_req,
                          conn_request->cm_idx, conn_request->dev_name,
                          ep_init_flags, sa_data_version, remote_addr,
                          addr_indices);
    if (status != UCS_OK) {
        ucs_warn("server ep %p failed to connect CM lane on device %s, "
                 "tl_bitmap " UCT_TL_BITMAP_FMT ", status %s",
                 ep, conn_request->dev_name, UCT_TL_BITMAP_ARG(&tl_bitmap),
                 ucs_status_string(status));
        goto err_destroy_ep;
    }

    ucp_ep_update_remote_id(ep, sa_data->ep_id);

    if (conn_request->listener->accept_cb == NULL) {
        goto out_free_request;
    } else {
        conn_request->ep = ep;
        ucp_listener_schedule_accept_cb(conn_request);
        goto out;
    }

err_destroy_ep:
    ucp_ep_destroy_internal(ep);
out_free_request:
    ucs_free(conn_request->remote_dev_addr);
    ucs_free(conn_request);
out:
    UCS_ASYNC_BLOCK(&worker->async);
    listener->conn_reqs--;
    UCS_ASYNC_UNBLOCK(&worker->async);
    if (status == UCS_OK) {
        *ep_p = ep;
    }

    return status;
}

static ucs_status_t
ucp_ep_server_init_priv_data(ucp_ep_h ep, const char *dev_name,
                             const void **data_buf_p, size_t *data_buf_size_p,
                             unsigned ep_init_flags,
                             ucp_object_version_t sa_data_version,
                             unsigned max_num_paths)
{
    ucp_wireup_ep_t *cm_wireup_ep = ucp_ep_get_cm_wireup_ep(ep);
    ucp_worker_h worker           = ep->worker;
    unsigned ep_pack_flags        = ucp_wireup_cm_address_pack_flags(worker,
                                                                     cm_wireup_ep,
                                                                     ep_init_flags);
    ucp_tl_bitmap_t tl_bitmap;
    ucp_tl_bitmap_t ctx_tl_bitmap;
    ucs_status_t status;

    UCS_ASYNC_BLOCK(&worker->async);

    UCP_EP_CM_CALLBACK_ENTER(ep, ucp_ep_get_cm_uct_ep(ep),
                             {
                                 status = UCS_ERR_NOT_CONNECTED;
                                 goto out;
                             });

    ucp_ep_get_tl_bitmap(&ucp_ep_config(ep)->key, &tl_bitmap);

    ucp_context_dev_tl_bitmap(worker->context, dev_name, &ctx_tl_bitmap);
    ucp_tl_bitmap_validate(&tl_bitmap, &ctx_tl_bitmap);

    status = ucp_cm_ep_priv_data_pack(ep, &tl_bitmap, UCS_LOG_LEVEL_ERROR,
                                      sa_data_version, (void**)data_buf_p,
                                      data_buf_size_p, ep_pack_flags,
                                      max_num_paths);

out:
    UCS_ASYNC_UNBLOCK(&worker->async);
    return status;
}

/*
 * The main thread progress part of connection establishment on server side
 */
static unsigned ucp_cm_server_conn_notify_progress(void *arg)
{
    ucp_ep_h ucp_ep = arg;

    UCS_ASYNC_BLOCK(&ucp_ep->worker->async);

    ucs_assert(ucp_ep->flags & UCP_EP_FLAG_LOCAL_CONNECTED);

    if (ucp_ep->worker->context->config.ext.cm_use_all_devices) {
        ucp_wireup_update_flags(ucp_ep, UCP_WIREUP_EP_FLAG_REMOTE_CONNECTED);
        ucp_wireup_send_pre_request(ucp_ep);
    } else {
        ucp_wireup_remote_connected(ucp_ep);
    }

    UCS_ASYNC_UNBLOCK(&ucp_ep->worker->async);
    return 1;
}

/*
 * Async callback on a server side which notifies that client is connected.
 */
static void ucp_cm_server_conn_notify_cb(
        uct_ep_h uct_cm_ep, void *arg,
        const uct_cm_ep_server_conn_notify_args_t *notify_args)
{
    ucp_ep_h ucp_ep = arg;
    ucs_status_t status;

    ucs_assert_always(notify_args->field_mask &
                      UCT_CM_EP_SERVER_CONN_NOTIFY_ARGS_FIELD_STATUS);

    status = notify_args->status;
    ucp_ep_update_flags(ucp_ep, UCP_EP_FLAG_SERVER_NOTIFY_CB, 0);
    ucs_trace("ep %p flags 0x%x: notify callback invoked, status %s", ucp_ep,
              ucp_ep->flags, ucs_status_string(status));

    UCP_EP_CM_CALLBACK_ENTER(ucp_ep, uct_cm_ep, return);

    if (status == UCS_OK) {
        ucs_callbackq_add_oneshot(&ucp_ep->worker->uct->progress_q, ucp_ep,
                                  ucp_cm_server_conn_notify_progress, ucp_ep);
        ucp_worker_signal_internal(ucp_ep->worker);
    } else {
        /* if reject is arrived on server side, then UCT does something wrong */
        ucs_assert(status != UCS_ERR_REJECTED);
        ucp_ep_set_failed_schedule(ucp_ep, ucp_ep_get_cm_lane(ucp_ep), status);
    }
}

ucs_status_t
ucp_ep_cm_connect_server_lane(ucp_ep_h ep, uct_listener_h uct_listener,
                              uct_conn_request_h uct_conn_req,
                              ucp_rsc_index_t cm_idx, const char *dev_name,
                              unsigned ep_init_flags,
                              ucp_object_version_t sa_data_version,
                              const ucp_unpacked_address_t *remote_address,
                              const unsigned *addr_indices)
{
    ucp_worker_h worker    = ep->worker;
    ucp_lane_index_t lane  = ucp_ep_get_cm_lane(ep);
    unsigned max_num_paths = 0;
    uct_ep_params_t uct_ep_params;
    const ucp_address_entry_t *ae;
    uct_ep_h uct_ep;
    ucs_status_t status;

    ucs_assert(ucp_ep_get_lane(ep, lane) == NULL);

    ucp_unpacked_address_for_each(ae, remote_address) {
        max_num_paths = ucs_max(max_num_paths, ae->dev_num_paths);
    }

    /* Address should not be empty and contain transport addresses of a single
     * device.
     */
    ucs_assert(max_num_paths > 0);

    /* TODO: split CM and wireup lanes */
    status = ucp_wireup_ep_create(ep, &uct_ep);
    if (status != UCS_OK) {
        ucs_warn("server ep %p failed to create wireup CM lane, status %s",
                 ep, ucs_status_string(status));
        uct_listener_reject(uct_listener, uct_conn_req);
        goto err;
    }

    ucp_ep_set_lane(ep, lane, uct_ep);
    ep->ext->cm_idx = cm_idx;

    /* create a server side CM endpoint */
    ucs_trace("server ep %p: uct_ep[%d], worker %p, cm_idx=%d, cm=%s",
              ep, lane, worker, cm_idx,
              ucp_context_cm_name(worker->context, cm_idx));
    uct_ep_params.field_mask = UCT_EP_PARAM_FIELD_CM                        |
                               UCT_EP_PARAM_FIELD_CONN_REQUEST              |
                               UCT_EP_PARAM_FIELD_USER_DATA                 |
                               UCT_EP_PARAM_FIELD_SOCKADDR_CB_FLAGS         |
                               UCT_EP_PARAM_FIELD_SOCKADDR_NOTIFY_CB_SERVER |
                               UCT_EP_PARAM_FIELD_SOCKADDR_DISCONNECT_CB    |
                               UCT_EP_PARAM_FIELD_PRIV_DATA                 |
                               UCT_EP_PARAM_FIELD_PRIV_DATA_LENGTH;

    uct_ep_params.cm                 = worker->cms[cm_idx].cm;
    uct_ep_params.user_data          = ep;
    uct_ep_params.conn_request       = uct_conn_req;
    uct_ep_params.sockaddr_cb_flags  = UCT_CB_FLAG_ASYNC;
    uct_ep_params.sockaddr_cb_server = ucp_cm_server_conn_notify_cb;
    uct_ep_params.disconnect_cb      = ucp_cm_disconnect_cb;
    status = ucp_ep_server_init_priv_data(ep, dev_name,
                                          &uct_ep_params.private_data,
                                          &uct_ep_params.private_data_length,
                                          ep_init_flags, sa_data_version,
                                          max_num_paths);
    if (status != UCS_OK) {
        goto err;
    }

    status = uct_ep_create(&uct_ep_params, &uct_ep);
    ucs_free((void*)uct_ep_params.private_data);
    if (status != UCS_OK) {
        goto err;
    }

    ucp_wireup_ep_set_next_ep(ucp_ep_get_lane(ep, lane), uct_ep,
                              UCP_NULL_RESOURCE);
    ucp_ep_update_flags(ep, UCP_EP_FLAG_LOCAL_CONNECTED, 0);
    return UCS_OK;

err:
    /* coverity[leaked_storage] (uct_ep) */
    return status;
}

static int
ucp_cm_connect_progress_remove_filter(const ucs_callbackq_elem_t *elem,
                                      void *arg)
{
    ucp_cm_client_connect_progress_arg_t *client_connect_arg;

    if (elem->cb == ucp_cm_client_connect_progress) {
        client_connect_arg = elem->arg;
        if (client_connect_arg->ucp_ep == arg) {
            ucp_cm_client_connect_prog_arg_free(client_connect_arg);
            return 1;
        }
    } else if (((elem->cb == ucp_cm_server_conn_notify_progress) ||
                (elem->cb == ucp_cm_client_uct_connect_progress) ||
                (elem->cb == ucp_cm_client_try_next_cm_progress)) &&
               (elem->arg == arg)) {
        return 1;
    }

    return 0;
}

void ucp_ep_cm_disconnect_cm_lane(ucp_ep_h ucp_ep)
{
    uct_ep_h uct_cm_ep = ucp_ep_get_cm_uct_ep(ucp_ep);
    ucs_status_t status;

    ucs_assert_always(uct_cm_ep != NULL);
    /* No reason to try disconnect twice */
    ucs_assert(!(ucp_ep->flags & UCP_EP_FLAG_DISCONNECTED_CM_LANE));
    ucs_assert(!(ucp_ep->flags & UCP_EP_FLAG_FAILED));
    ucs_assert(ucp_ep_is_cm_local_connected(ucp_ep));

    ucp_ep_update_flags(ucp_ep, UCP_EP_FLAG_DISCONNECTED_CM_LANE,
                        UCP_EP_FLAG_LOCAL_CONNECTED);

    /* Remove the client's connect_progress or the server's connect notify
     * callbacks from the callbackq to make sure it won't be invoked after
     * CM lane has already been disconnected */
    ucs_callbackq_remove_oneshot(&ucp_ep->worker->uct->progress_q, ucp_ep,
                                 ucp_cm_connect_progress_remove_filter, ucp_ep);

    /* this will invoke @ref ucp_cm_disconnect_cb on remote side */
    status = uct_ep_disconnect(uct_cm_ep, 0);
    if (status != UCS_OK) {
        ucs_diag("failed to disconnect CM lane %p of ep %p, %s", ucp_ep,
                 uct_cm_ep, ucs_status_string(status));
    }
}

ucp_request_t* ucp_ep_cm_close_request_get(ucp_ep_h ep, const ucp_request_param_t *param)
{
    ucp_request_t *request = ucp_request_get_param(ep->worker, param, {return NULL;});

    if (request == NULL) {
        ucs_error("failed to allocate close request for ep %p", ep);
        return NULL;
    }

    request->status  = UCS_OK;
    request->flags   = 0;
    request->send.ep = ep;
    request->send.flush.uct_flags = UCT_FLUSH_FLAG_LOCAL;

    ucp_request_set_send_callback_param(param, request, send);

    return request;
}

static int ucp_cm_progress_remove_filter(const ucs_callbackq_elem_t *elem,
                                         void *arg)
{
    return ucp_cm_connect_progress_remove_filter(elem, arg) ||
           ((elem->cb == ucp_ep_cm_disconnect_progress) && (elem->arg == arg));
}

void ucp_ep_cm_slow_cbq_cleanup(ucp_ep_h ep)
{
    ucs_callbackq_remove_oneshot(&ep->worker->uct->progress_q, ep,
                                 ucp_cm_progress_remove_filter, ep);
}
