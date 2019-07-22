/**
 * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 * Copyright (C) NVIDIA Corporation. 2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_SOCKCM_IFACE_H
#define UCT_SOCKCM_IFACE_H

#include "sockcm_def.h"
#include "sockcm_md.h"

typedef struct uct_sockcm_iface_config {
    uct_iface_config_t       super;
    unsigned                 backlog;
} uct_sockcm_iface_config_t;

struct uct_sockcm_iface {
    uct_base_iface_t                     super;

    int                                  sock_id;
    int                                  listen_fd;

    uint8_t                              is_server;
    /* Fields used only for server side */
    void                                 *conn_request_arg;
    uct_sockaddr_conn_request_callback_t conn_request_cb;
    uint32_t                             cb_flags;

    /* Field used only for client side */
    ucs_list_link_t                      pending_eps_list;
    ucs_list_link_t                      used_sock_ids_list;
};

void uct_sockcm_iface_client_start_next_ep(uct_sockcm_iface_t *iface);

extern uct_md_component_t uct_sockcm_mdc;

#endif
