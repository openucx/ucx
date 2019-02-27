/**
 * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_RDMACM_IFACE_H
#define UCT_RDMACM_IFACE_H

#include "rdmacm_def.h"
#include "rdmacm_md.h"

#define UCT_RDMACM_MAX_CONN_PRIV \
        (UCT_RDMACM_UDP_PRIV_DATA_LEN) - (sizeof(uct_rdmacm_priv_data_hdr_t))


#define UCT_RDMACM_CM_MAX_CONN_PRIV \
        (UCT_RDMACM_TCP_PRIV_DATA_LEN) - (sizeof(uct_rdmacm_priv_data_hdr_t))

typedef struct uct_rdmacm_iface_config {
    uct_iface_config_t       super;
    unsigned                 backlog;
    unsigned                 cm_id_quota;
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

    /** Field used only for client side */
    ucs_list_link_t                      pending_eps_list;
    ucs_list_link_t                      used_cm_ids_list;
    int                                  cm_id_quota; /* num of cm_ids in the quota*/

    struct {
        double                           addr_resolve_timeout;
    } config;
};

void uct_rdmacm_iface_client_start_next_ep(uct_rdmacm_iface_t *iface);

extern uct_md_component_t uct_rdmacm_mdc;

#endif
