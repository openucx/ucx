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
    return UCS_ERR_NOT_IMPLEMENTED;
}

void uct_listener_destroy(uct_listener_h listener){}
