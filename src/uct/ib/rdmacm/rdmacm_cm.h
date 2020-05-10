/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_RDMACM_CM_H
#define UCT_RDMACM_CM_H

#include <uct/base/uct_cm.h>
#include "rdmacm_def.h"


/**
 * An rdmacm connection manager
 */
typedef struct uct_rdmacm_cm {
    uct_cm_t                  super;
    struct rdma_event_channel *ev_ch;
} uct_rdmacm_cm_t;

UCS_CLASS_DECLARE_NEW_FUNC(uct_rdmacm_cm_t, uct_cm_t, uct_component_h,
                           uct_worker_h, const uct_cm_config_t *);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_rdmacm_cm_t, uct_cm_t);

static UCS_F_ALWAYS_INLINE ucs_async_context_t *
uct_rdmacm_cm_get_async(uct_rdmacm_cm_t *cm)
{
    uct_priv_worker_t *wpriv = ucs_derived_of(cm->super.iface.worker,
                                              uct_priv_worker_t);

    return wpriv->async;
}

ucs_status_t uct_rdmacm_cm_destroy_id(struct rdma_cm_id *id);

ucs_status_t uct_rdmacm_cm_ack_event(struct rdma_cm_event *event);

ucs_status_t uct_rdmacm_cm_reject(struct rdma_cm_id *id);

#endif
