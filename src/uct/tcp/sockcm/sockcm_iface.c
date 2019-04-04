/**
 * Copyright (C) Mellanox Technologies Ltd. 2017-2019.  ALL RIGHTS RESERVED.
 * Copyright (C) NVIDIA Corporation. 2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "sockcm_iface.h"
#include "sockcm_ep.h"
#include <uct/base/uct_worker.h>
#include <ucs/sys/string.h>
#include <ucs/sys/sock.h>

enum uct_sockcm_process_event_flags {
    UCT_SOCKCM_PROCESS_EVENT_DESTROY_SOCK_ID_FLAG = UCS_BIT(0),
    UCT_SOCKCM_PROCESS_EVENT_ACK_EVENT_FLAG       = UCS_BIT(1)
};

static ucs_config_field_t uct_sockcm_iface_config_table[] = {
    {"BACKLOG", "1024",
     "Maximum number of pending connections for a listening socket.",
     ucs_offsetof(uct_sockcm_iface_config_t, backlog), UCS_CONFIG_TYPE_UINT},

    {NULL}
};

static UCS_CLASS_DECLARE_DELETE_FUNC(uct_sockcm_iface_t, uct_iface_t);

static ucs_status_t uct_sockcm_iface_query(uct_iface_h tl_iface,
                                           uct_iface_attr_t *iface_attr)
{
    memset(iface_attr, 0, sizeof(uct_iface_attr_t));

    iface_attr->iface_addr_len  = sizeof(ucs_sock_addr_t);
    iface_attr->device_addr_len = 0;
    iface_attr->cap.flags       = UCT_IFACE_FLAG_CONNECT_TO_SOCKADDR    |
                                  UCT_IFACE_FLAG_CB_ASYNC               |
                                  UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE;

    return UCS_OK;
}

static ucs_status_t uct_sockcm_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *iface_addr)
{
    ucs_sock_addr_t *sockcm_addr = (ucs_sock_addr_t *)iface_addr;

    sockcm_addr->addr    = NULL;
    sockcm_addr->addrlen = 0;
    return UCS_OK;
}

static ucs_status_t uct_sockcm_iface_accept(uct_iface_h tl_iface,
                                            uct_conn_request_h conn_request)
{
    return UCS_ERR_NOT_IMPLEMENTED;
}

static ucs_status_t uct_sockcm_iface_reject(uct_iface_h tl_iface,
                                            uct_conn_request_h conn_request)
{
    return UCS_ERR_NOT_IMPLEMENTED;
}

static ucs_status_t uct_sockcm_ep_flush(uct_ep_h tl_ep, unsigned flags,
                                        uct_completion_t *comp)
{
    return UCS_ERR_NOT_IMPLEMENTED;
}


static uct_iface_ops_t uct_sockcm_iface_ops = {
    .ep_create                = UCS_CLASS_NEW_FUNC_NAME(uct_sockcm_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_sockcm_ep_t),
    .ep_flush                 = uct_sockcm_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .ep_pending_purge         = ucs_empty_function,
    .iface_accept             = uct_sockcm_iface_accept,
    .iface_reject             = uct_sockcm_iface_reject,
    .iface_progress_enable    = (void*)ucs_empty_function_return_success,
    .iface_progress_disable   = (void*)ucs_empty_function_return_success,
    .iface_progress           = ucs_empty_function_return_zero,
    .iface_flush              = uct_base_iface_flush,
    .iface_fence              = uct_base_iface_fence,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_sockcm_iface_t),
    .iface_query              = uct_sockcm_iface_query,
    .iface_is_reachable       = (void*)ucs_empty_function_return_zero,
    .iface_get_device_address = (void*)ucs_empty_function_return_success,
    .iface_get_address        = uct_sockcm_iface_get_address
};

void uct_sockcm_iface_client_start_next_ep(uct_sockcm_iface_t *iface)
{
    ucs_trace("sockcm_iface_client_start_next_ep not implemented");
}

static UCS_CLASS_INIT_FUNC(uct_sockcm_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &uct_sockcm_iface_ops, md, worker,
                              params, tl_config
                              UCS_STATS_ARG((params->field_mask &
                                             UCT_IFACE_PARAM_FIELD_STATS_ROOT) ?
                                            params->stats_root : NULL)
                              UCS_STATS_ARG(UCT_SOCKCM_TL_NAME));

    ucs_list_head_init(&self->pending_eps_list);
    ucs_list_head_init(&self->used_sock_ids_list);

    return UCS_OK;

}

static UCS_CLASS_CLEANUP_FUNC(uct_sockcm_iface_t)
{
}

UCS_CLASS_DEFINE(uct_sockcm_iface_t, uct_base_iface_t);
static UCS_CLASS_DEFINE_NEW_FUNC(uct_sockcm_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const uct_iface_params_t *,
                                 const uct_iface_config_t *);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_sockcm_iface_t, uct_iface_t);

static ucs_status_t uct_sockcm_query_tl_resources(uct_md_h md,
                                                  uct_tl_resource_desc_t **resource_p,
                                                  unsigned *num_resources_p)
{
    *num_resources_p = 0;
    *resource_p      = NULL;
    return UCS_OK;
}

UCT_TL_COMPONENT_DEFINE(uct_sockcm_tl,
                        uct_sockcm_query_tl_resources,
                        uct_sockcm_iface_t,
                        UCT_SOCKCM_TL_NAME,
                        "SOCKCM_",
                        uct_sockcm_iface_config_table,
                        uct_sockcm_iface_config_t);
UCT_MD_REGISTER_TL(&uct_sockcm_mdc, &uct_sockcm_tl);
