/**
 * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_RDMACM_IFACE_H
#define UCT_RDMACM_IFACE_H

#include "rdmacm_def.h"
#include "rdmacm_md.h"


typedef struct uct_rdmacm_iface_config {
    uct_iface_config_t       super;
    unsigned                 backlog;
} uct_rdmacm_iface_config_t;


struct uct_rdmacm_iface {
    uct_base_iface_t                     super;

    struct rdma_cm_id                    *cm_id;
    struct rdma_event_channel            *event_ch;

    uint8_t                              is_server;
    /** Fields used only for server side */
    void                                 *conn_request_arg;
    uct_sockaddr_conn_request_callback_t conn_request_cb;
    uint32_t                             cb_flags;

    uct_rdmacm_ep_t                      *ep;

    struct {
        double                           addr_resolve_timeout;
    } config;
};


extern uct_md_component_t uct_rdmacm_mdc;

#endif
