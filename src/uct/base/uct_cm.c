/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "uct_cm.h"

#include <ucs/sys/math.h>
#include <uct/base/uct_md.h>
#include <ucs/sys/string.h>


ucs_config_field_t uct_cm_config_table[] = {
  {NULL}
};

ucs_status_t uct_cm_open(uct_component_h component, uct_worker_h worker,
                         const uct_cm_config_t *config, uct_cm_h *cm_p)
{
    return component->cm_open(component, worker, config, cm_p);
}

void uct_cm_close(uct_cm_h cm)
{
    cm->ops->close(cm);
}

ucs_status_t uct_cm_query(uct_cm_h cm, uct_cm_attr_t *cm_attr)
{
    return cm->ops->cm_query(cm, cm_attr);
}

ucs_status_t uct_cm_config_read(uct_component_h component,
                                const char *env_prefix, const char *filename,
                                uct_cm_config_t **config_p)
{
    uct_config_bundle_t *bundle = NULL;
    ucs_status_t status;

    status = uct_config_read(&bundle, component->cm_config.table,
                             component->cm_config.size, env_prefix,
                             component->cm_config.prefix);
    if (status != UCS_OK) {
        ucs_error("failed to read CM configuration");
        return status;
    }

    *config_p = (uct_cm_config_t*) bundle->data;
    /* coverity[leaked_storage] */
    return UCS_OK;
}

ucs_status_t uct_cm_ep_pack_cb(uct_cm_base_ep_t *cep, void *arg,
                               const uct_cm_ep_priv_data_pack_args_t *pack_args,
                               void *priv_data, size_t priv_data_max,
                               size_t *priv_data_ret)
{
    ucs_status_t status = UCS_OK;
    ssize_t ret;

    ret = cep->priv_pack_cb(arg, pack_args, priv_data);
    if (ret < 0) {
        ucs_assert(ret > UCS_ERR_LAST);
        status = (ucs_status_t)ret;
        ucs_error("private data pack function failed with error: %s",
                  ucs_status_string(status));
        goto out;
    } else if (ret > priv_data_max) {
        status = UCS_ERR_EXCEEDS_LIMIT;
        ucs_error("private data pack function returned %zd (max: %zu)",
                  ret, priv_data_max);
        goto out;
    }

    *priv_data_ret = ret;
out:
    return status;
}

void uct_cm_ep_disconnect_cb(uct_cm_base_ep_t *cep)
{
    cep->disconnect_cb(&cep->super.super, cep->user_data);
}

void uct_cm_ep_client_connect_cb(uct_cm_base_ep_t *cep,
                                 uct_cm_remote_data_t *remote_data,
                                 ucs_status_t status)
{
    uct_cm_ep_client_connect_args_t connect_args;

    connect_args.field_mask  = UCT_CM_EP_CLIENT_CONNECT_ARGS_FIELD_REMOTE_DATA |
                               UCT_CM_EP_CLIENT_CONNECT_ARGS_FIELD_STATUS;
    connect_args.remote_data = remote_data;
    connect_args.status      = status;

    cep->client.connect_cb(&cep->super.super, cep->user_data, &connect_args);
}

void uct_cm_ep_server_conn_notify_cb(uct_cm_base_ep_t *cep, ucs_status_t status)
{
    uct_cm_ep_server_conn_notify_args_t notify_args;

    notify_args.field_mask = UCT_CM_EP_SERVER_CONN_NOTIFY_ARGS_FIELD_STATUS;
    notify_args.status     = status;

    cep->server.notify_cb(&cep->super.super, cep->user_data, &notify_args);
}

static ucs_status_t uct_cm_check_ep_params(const uct_ep_params_t *params)
{
    if (!(params->field_mask & UCT_EP_PARAM_FIELD_CM)) {
        ucs_error("UCT_EP_PARAM_FIELD_CM is not set. field_mask 0x%"PRIx64,
                  params->field_mask);
        return UCS_ERR_INVALID_PARAM;
    }

    if (!(params->field_mask & UCT_EP_PARAM_FIELD_SOCKADDR_CB_FLAGS) ||
        !(params->sockaddr_cb_flags & UCT_CB_FLAG_ASYNC)) {
        ucs_error("UCT_EP_PARAM_FIELD_SOCKADDR_CB_FLAGS and UCT_CB_FLAG_ASYNC "
                  "should be set. field_mask 0x%"PRIx64
                  ", sockaddr_cb_flags 0x%x",
                  params->field_mask, params->sockaddr_cb_flags);
        return UCS_ERR_UNSUPPORTED;
    }

    return UCS_OK;
}

ucs_status_t uct_cm_set_common_data(uct_cm_base_ep_t *ep,
                                    const uct_ep_params_t *params)
{
    ucs_status_t status;

    status = uct_cm_check_ep_params(params);
    if (status != UCS_OK) {
        return status;
    }

    status = UCT_CM_SET_CB(params, UCT_EP_PARAM_FIELD_SOCKADDR_PACK_CB,
                           ep->priv_pack_cb, params->sockaddr_pack_cb,
                           uct_cm_ep_priv_data_pack_callback_t,
                           ucs_empty_function_return_invalid_param);
    if (status != UCS_OK) {
        return status;
    }

    status = UCT_CM_SET_CB(params, UCT_EP_PARAM_FIELD_SOCKADDR_DISCONNECT_CB,
                           ep->disconnect_cb, params->disconnect_cb,
                           uct_ep_disconnect_cb_t, ucs_empty_function);
    if (status != UCS_OK) {
        return status;
    }

    ep->user_data = (params->field_mask & UCT_EP_PARAM_FIELD_USER_DATA) ?
                    params->user_data : NULL;

    return UCS_OK;
}

UCS_CLASS_INIT_FUNC(uct_cm_base_ep_t, const uct_ep_params_t *params)
{
    ucs_status_t status;

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &params->cm->iface);

    status = uct_cm_set_common_data(self, params);
    if (status != UCS_OK) {
        return status;
    }

    return UCS_OK;
}

UCS_CLASS_CLEANUP_FUNC(uct_cm_base_ep_t){}

UCS_CLASS_DEFINE(uct_cm_base_ep_t, uct_base_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_cm_base_ep_t, uct_base_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_cm_base_ep_t, uct_base_ep_t);


UCS_CLASS_INIT_FUNC(uct_listener_t, uct_cm_h cm)
{
    self->cm = cm;
    return UCS_OK;
}

UCS_CLASS_CLEANUP_FUNC(uct_listener_t){}

UCS_CLASS_DEFINE(uct_listener_t, void);
UCS_CLASS_DEFINE_NEW_FUNC(uct_listener_t, void, uct_cm_h);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_listener_t, void);

ucs_status_t uct_listener_create(uct_cm_h cm, const struct sockaddr *saddr,
                                 socklen_t socklen, const uct_listener_params_t *params,
                                 uct_listener_h *listener_p)
{
    if (!(params->field_mask & UCT_LISTENER_PARAM_FIELD_CONN_REQUEST_CB)) {
        return UCS_ERR_INVALID_PARAM;
    }

    return cm->ops->listener_create(cm, saddr, socklen, params, listener_p);
}

void uct_listener_destroy(uct_listener_h listener)
{
    listener->cm->ops->listener_destroy(listener);
}

ucs_status_t uct_listener_query(uct_listener_h listener,
                                uct_listener_attr_t *listener_attr)
{
    return listener->cm->ops->listener_query(listener, listener_attr);
}

ucs_status_t uct_listener_reject(uct_listener_h listener,
                                 uct_conn_request_h conn_request)
{
    return listener->cm->ops->listener_reject(listener, conn_request);
}


#ifdef ENABLE_STATS
static ucs_stats_class_t uct_cm_stats_class = {
    .name           = "rdmacm_cm",
    .num_counters   = 0
};
#endif

UCS_CLASS_INIT_FUNC(uct_cm_t, uct_cm_ops_t* ops, uct_iface_ops_t* iface_ops,
                    uct_worker_h worker, uct_component_h component)
{
    self->ops                     = ops;
    self->component               = component;
    self->iface.super.ops         = *iface_ops;
    self->iface.worker            = ucs_derived_of(worker, uct_priv_worker_t);

    self->iface.md                = NULL;
    self->iface.am->arg           = NULL;
    self->iface.am->flags         = 0;
    self->iface.am->cb            = (uct_am_callback_t)ucs_empty_function_return_unsupported;
    self->iface.am_tracer         = NULL;
    self->iface.am_tracer_arg     = NULL;
    self->iface.err_handler       = NULL;
    self->iface.err_handler_arg   = NULL;
    self->iface.err_handler_flags = 0;
    self->iface.prog.id           = UCS_CALLBACKQ_ID_NULL;
    self->iface.prog.refcount     = 0;
    self->iface.progress_flags    = 0;

    return UCS_STATS_NODE_ALLOC(&self->iface.stats, &uct_cm_stats_class,
                                ucs_stats_get_root(), "%s-%p", "iface",
                                self->iface);
}

UCS_CLASS_CLEANUP_FUNC(uct_cm_t)
{
    UCS_STATS_NODE_FREE(self->iface.stats);
}

UCS_CLASS_DEFINE(uct_cm_t, void);
UCS_CLASS_DEFINE_NEW_FUNC(uct_cm_t, void, uct_cm_ops_t*, uct_iface_ops_t*,
                          uct_worker_h, uct_component_h);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_cm_t, void);
