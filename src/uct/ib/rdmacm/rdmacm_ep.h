/**
 * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
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
    ucs_list_link_t                    list_elem;
    struct sockaddr                    *remote_addr;
    uint32_t                           cb_flags;
    int                                is_on_pending;
};

UCS_CLASS_DECLARE_NEW_FUNC(uct_rdmacm_ep_t, uct_ep_t, uct_iface_t*,
                           const ucs_sock_addr_t *,
                           uct_sockaddr_conn_reply_callback_t ,
                           void *, uint32_t, const void *, size_t);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_rdmacm_ep_t, uct_ep_t);

void uct_rdmacm_ep_set_failed(uct_iface_t *iface, uct_ep_h ep, ucs_status_t status);

#endif
