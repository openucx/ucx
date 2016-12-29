/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_TCP_MD_H
#define UCT_TCP_MD_H

#include <uct/base/uct_md.h>
#include <net/if.h>

#define UCT_TCP_NAME "tcp"


/**
 * TCP endpoint
 */
typedef struct uct_tcp_ep {
    uct_base_ep_t       super;
    int                 fd;
} uct_tcp_ep_t;


/**
 * TCP interface
 */
typedef struct uct_tcp_iface {
    uct_base_iface_t         super;      /* Parent class */
    int                      listen_fd;  /* Server socket */

    struct {
        struct sockaddr_in   ifaddr;
    } config;
} uct_tcp_iface_t;


/**
 * TCP interface configuration
 */
typedef struct uct_tcp_iface_config {
    uct_iface_config_t       super;
    int                      prefer_default;
    unsigned                 backlog;
    size_t                   sndbuf;
    size_t                   rcvbuf;
} uct_tcp_iface_config_t;


extern uct_md_component_t uct_tcp_md;
extern const char *uct_tcp_address_type_names[];

ucs_status_t uct_tcp_socket_create(int *fd_p);

ucs_status_t uct_tcp_socket_connect(int fd, const struct sockaddr_in *dest_addr);

int uct_tcp_netif_check(const char *if_name);

ucs_status_t uct_tcp_netif_inaddr(const char *if_name, struct sockaddr_in *ifaddr);

UCS_CLASS_DECLARE_NEW_FUNC(uct_tcp_ep_t, uct_ep_t, uct_iface_t *,
                           const uct_device_addr_t *, const uct_iface_addr_t *);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_tcp_ep_t, uct_ep_t);

#endif
