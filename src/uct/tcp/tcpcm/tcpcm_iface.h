/**
 * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_TCPCM_IFACE_H
#define UCT_TCPCM_IFACE_H

#include "tcpcm_def.h"
#include "tcpcm_md.h"

#define UCT_TCPCM_MAX_CONN_PRIV \
        (UCT_TCPCM_UDP_PRIV_DATA_LEN) - (sizeof(uct_tcpcm_priv_data_hdr_t))

typedef struct uct_tcpcm_iface_config {
    uct_iface_config_t       super;
    unsigned                 backlog;
    unsigned                 sock_id_quota;
} uct_tcpcm_iface_config_t;


struct uct_tcpcm_iface {
    uct_base_iface_t                     super;

    int                                  *sock_id;

    uint8_t                              is_server;
    /** Fields used only for server side */
    void                                 *conn_request_arg;
    uct_sockaddr_conn_request_callback_t conn_request_cb;
    uint32_t                             cb_flags;

    /** Field used only for client side */
    ucs_list_link_t                      pending_eps_list;
    ucs_list_link_t                      used_sock_ids_list;
    int                                  sock_id_quota; /* num of cm_ids in the quota*/

    /* *FIXME* // do we need resolve timeout in tcp?
    */
    struct {
        double                           addr_resolve_timeout;
    } config;
};

void uct_tcpcm_iface_client_start_next_ep(uct_tcpcm_iface_t *iface);

extern uct_md_component_t uct_tcpcm_mdc;

#endif
