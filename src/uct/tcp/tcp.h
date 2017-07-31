/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_TCP_MD_H
#define UCT_TCP_MD_H

#include <uct/base/uct_md.h>
#include <ucs/datastruct/khash.h>
#include <net/if.h>

#define UCT_TCP_NAME "tcp"


/** Hash of fd->rsock */
typedef struct uct_tcp_recv_sock uct_tcp_recv_sock_t;
KHASH_MAP_INIT_INT64(uct_tcp_fd_hash, uct_tcp_recv_sock_t*);


/** How many events to wait for in epoll_wait */
#define UCT_TCP_MAX_EVENTS        32


/**
 * TCP active message header
 */
typedef struct uct_tcp_am_hdr {
    uint16_t                      am_id;
    uint16_t                      length;
    UCS_DEBUG_DATA(uint32_t       msn);
} UCS_S_PACKED uct_tcp_am_hdr_t;


/**
 * TCP endpoint
 */
typedef struct uct_tcp_ep {
    uct_base_ep_t                 super;
    int                           fd;             /* Socket file descriptor */
    UCS_DEBUG_DATA(uint32_t       msn);           /* Message sequence number (for debug) */
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
    int                           recv_epfd;      /* event poll set of recv sockets */
    uint32_t                      recv_sock_count;/* how many receive sockets */
    uct_recv_desc_t               release_desc;   /* active message release callback */

    struct {
        struct sockaddr_in        ifaddr;         /* Network address */
        struct sockaddr_in        netmask;        /* Network address mask */
        size_t                    max_bcopy;      /* Maximal bcopy size */
        int                       prefer_default; /* prefer default gateway */
        ptrdiff_t                 am_hdr_offset;  /* offset to receive header */
        ptrdiff_t                 headroom_offset;/* offset to receive headroom */
        unsigned                  max_poll;       /* number of events to poll per socket*/
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
    unsigned                      max_poll;
    int                           sockopt_nodelay;
} uct_tcp_iface_config_t;


/**
 * TCP received active message
 */
typedef void uct_tcp_am_desc_t;


/**
 * TCP receive socket wrapper/
 */
struct uct_tcp_recv_sock {
    uct_tcp_am_desc_t            *desc;  /* Partial received data (can be NULL) */
    size_t                       offset; /* Offset to next data receive */
};


extern uct_md_component_t uct_tcp_md;
extern const char *uct_tcp_address_type_names[];

ucs_status_t uct_tcp_socket_create(int *fd_p);

ucs_status_t uct_tcp_socket_connect(int fd, const struct sockaddr_in *dest_addr);

int uct_tcp_netif_check(const char *if_name);

ucs_status_t uct_tcp_netif_caps(const char *if_name, double *latency_p,
                                double *bandwidth_p);

ucs_status_t uct_tcp_netif_inaddr(const char *if_name, struct sockaddr_in *ifaddr,
                                  struct sockaddr_in *netmask);

ucs_status_t uct_tcp_netif_is_default(const char *if_name, int *result_p);

ucs_status_t uct_tcp_send(int fd, const void *data, size_t *length_p);

ucs_status_t uct_tcp_recv(int fd, void *data, size_t *length_p);

unsigned uct_tcp_iface_progress(uct_iface_h tl_iface);

ucs_status_t uct_tcp_iface_set_sockopt(uct_tcp_iface_t *iface, int fd);

ucs_status_t uct_tcp_iface_connection_accepted(uct_tcp_iface_t *iface, int fd);

void uct_tcp_iface_recv_cleanup(uct_tcp_iface_t *iface);

UCS_CLASS_DECLARE_NEW_FUNC(uct_tcp_ep_t, uct_ep_t, uct_iface_t *,
                           const uct_device_addr_t *, const uct_iface_addr_t *);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_tcp_ep_t, uct_ep_t);

ssize_t uct_tcp_ep_am_bcopy(uct_ep_h uct_ep, uint8_t am_id,
                            uct_pack_callback_t pack_cb, void *arg,
                            unsigned flags);


static inline uct_tcp_am_hdr_t *
uct_tcp_desc_hdr(uct_tcp_iface_t *iface, uct_tcp_am_desc_t *desc)
{
    return (void*)desc + iface->config.am_hdr_offset;
}

#endif
