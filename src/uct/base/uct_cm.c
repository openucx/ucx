/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "uct_cm.h"

#include <ucs/sys/math.h>
#include <uct/base/uct_md.h>


ucs_status_t uct_cm_open(uct_component_h component, uct_worker_h worker,
                         uct_cm_h *cm_p)
{
    return component->cm_open(component, worker, cm_p);
}

void uct_cm_close(uct_cm_h cm)
{
    cm->ops->close(cm);
}

ucs_status_t uct_cm_query(uct_cm_h cm, uct_cm_attr_t *cm_attr)
{
    return cm->ops->cm_query(cm, cm_attr);
}

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

ucs_status_t uct_listener_reject(uct_listener_h listener,
                                 uct_conn_request_h conn_request)
{
    return listener->cm->ops->listener_reject(listener, conn_request);
}

UCS_CLASS_INIT_FUNC(uct_cm_t, uct_cm_ops_t* ops, uct_component_h component)
{
    self->ops       = ops;
    self->component = component;
    return UCS_OK;
}

UCS_CLASS_CLEANUP_FUNC(uct_cm_t){}

UCS_CLASS_DEFINE(uct_cm_t, void);
UCS_CLASS_DEFINE_NEW_FUNC(uct_cm_t, void, uct_cm_ops_t*, uct_component_h);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_cm_t, void);
