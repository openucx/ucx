/**
 * Copyright (C) Mellanox Technologies Ltd. 2017-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_RDMACM_EP_H
#define UCT_RDMACM_EP_H

#include "rdmacm_iface.h"


typedef struct uct_rdmacm_ep_op uct_rdmacm_ep_op_t;

struct uct_rdmacm_ep_op {
    ucs_queue_elem_t    queue_elem;
    uct_completion_t    *user_comp;
};


struct uct_rdmacm_ep {
    uct_base_ep_t                       super;
    uct_cm_ep_priv_data_pack_callback_t pack_cb;
    void                                *pack_cb_arg;
    uint32_t                            pack_cb_flags;
    int                                 is_on_pending;

    pthread_mutex_t                     ops_mutex;  /* guards ops and status */
    ucs_queue_head_t                    ops;
    ucs_status_t                        status;     /* client EP status */

    ucs_list_link_t                     list_elem;  /* for the pending_eps_list */
    struct sockaddr_storage             remote_addr;
    uct_worker_cb_id_t                  slow_prog_id;
    uct_rdmacm_ctx_t                    *cm_id_ctx;
};

UCS_CLASS_DECLARE_NEW_FUNC(uct_rdmacm_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_rdmacm_ep_t, uct_ep_t);

void uct_rdmacm_ep_set_failed(uct_iface_t *iface, uct_ep_h ep, ucs_status_t status);

void uct_rdmacm_ep_invoke_completions(uct_rdmacm_ep_t *ep, ucs_status_t status);

#endif
