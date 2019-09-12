/**
 * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 * Copyright (C) NVIDIA Corporation. 2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_SOCKCM_IFACE_H
#define UCT_SOCKCM_IFACE_H

#include "sockcm_def.h"
#include "sockcm_md.h"

#define UCT_SOCKCM_MAX_CONN_PRIV \
        (UCT_SOCKCM_PRIV_DATA_LEN - sizeof(ssize_t))


typedef enum uct_sockcm_iface_notify {
    UCT_SOCKCM_IFACE_NOTIFY_ACCEPT = 0,
    UCT_SOCKCM_IFACE_NOTIFY_REJECT
} uct_sockcm_iface_notify_t;

typedef struct uct_sockcm_iface_config {
    uct_iface_config_t       super;
    unsigned                 backlog;
} uct_sockcm_iface_config_t;

struct uct_sockcm_iface {
    uct_base_iface_t                     super;

    int                                  listen_fd;

    uint8_t                              is_server;
    /* Fields used only for server side */
    void                                 *conn_request_arg;
    uct_sockaddr_conn_request_callback_t conn_request_cb;
    uint32_t                             cb_flags;

    /* Field used only for client side */
    ucs_list_link_t                      used_sock_ids_list;
};
#endif
