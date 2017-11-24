/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_TCP_MD_H
#define UCT_TCP_MD_H

#include <uct/base/uct_md.h>
#include <ucs/datastruct/khash.h>
#include <ucs/sys/sys.h>
#include <net/if.h>

#define UCT_TCP_NAME "tcp"


/** Hash of fd->rsock */
typedef struct uct_tcp_recv_sock uct_tcp_recv_sock_t;
KHASH_MAP_INIT_INT64(uct_tcp_fd_hash, uct_tcp_recv_sock_t*);


/**
 * TCP endpoint
 */
typedef struct uct_tcp_ep {
    uct_base_ep_t                 super;
    int                           fd;             /* Socket file descriptor */
} uct_tcp_ep_t;


/**
 * TCP interface
 */
typedef struct uct_tcp_iface {
    uct_base_iface_t              super;          /* Parent class */
    ucs_mpool_t                   mp;             /* Memory pool for TX/RX buffers */
    int                           listen_fd;      /* Server socket */
    khash_t(uct_tcp_fd_hash)      fd_hash;        /* Hash table of all FDs */
    char                          if_name[IFNAMSIZ];/* Network interface name */

    struct {
        struct sockaddr_in        ifaddr;         /* Network address */
        struct sockaddr_in        netmask;        /* Network address mask */
        size_t                    max_bcopy;      /* Maximal bcopy size */
        int                       prefer_default; /* prefer default gateway */
        ptrdiff_t                 am_hdr_offset;  /* offset to receive header */
        ptrdiff_t                 headroom_offset;/* offset to receive headroom */
    } config;

    struct {
        int                       nodelay;        /* TCP_NODELAY */
    } sockopt;
} uct_tcp_iface_t;


/**
 * TCP interface configuration
 */
typedef struct uct_tcp_iface_config {
    uct_iface_config_t            super;
    int                           prefer_default;
    unsigned                      backlog;
    int                           sockopt_nodelay;
} uct_tcp_iface_config_t;


/**
 * TCP receive socket wrapper/
 */
struct uct_tcp_recv_sock {
};


extern uct_md_component_t uct_tcp_md;
extern const char *uct_tcp_address_type_names[];

ucs_status_t uct_tcp_socket_connect(int fd, const struct sockaddr_in *dest_addr);

ucs_status_t uct_tcp_netif_caps(const char *if_name, double *latency_p,
                                double *bandwidth_p);

ucs_status_t uct_tcp_netif_inaddr(const char *if_name, struct sockaddr_in *ifaddr,
                                  struct sockaddr_in *netmask);

ucs_status_t uct_tcp_netif_is_default(const char *if_name, int *result_p);

ucs_status_t uct_tcp_iface_set_sockopt(uct_tcp_iface_t *iface, int fd);

ucs_status_t uct_tcp_iface_connection_accepted(uct_tcp_iface_t *iface, int fd);

void uct_tcp_iface_recv_cleanup(uct_tcp_iface_t *iface);

UCS_CLASS_DECLARE_NEW_FUNC(uct_tcp_ep_t, uct_ep_t, uct_iface_t *,
                           const uct_device_addr_t *, const uct_iface_addr_t *);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_tcp_ep_t, uct_ep_t);

#endif
