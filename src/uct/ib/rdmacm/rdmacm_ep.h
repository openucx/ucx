/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_RDMACM_EP_H
#define UCT_RDMACM_EP_H

#include "rdmacm_iface.h"

struct uct_rdmacm_ep {
    uct_base_ep_t                      super;
    void                               *priv_data;
    uct_sockaddr_conn_reply_callback_t conn_reply_cb;
    void                               *conn_reply_arg;
    ucs_queue_elem_t                   queue;
    struct sockaddr                    *remote_addr;
};

UCS_CLASS_DECLARE_NEW_FUNC(uct_rdmacm_ep_t, uct_ep_t, uct_iface_t*,
                           const ucs_sock_addr_t *,
                           uct_sockaddr_conn_reply_callback_t ,
                           void *, uint32_t, const void *, size_t);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_rdmacm_ep_t, uct_ep_t);

#endif
