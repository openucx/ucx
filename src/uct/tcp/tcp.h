/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_TCP_MD_H
#define UCT_TCP_MD_H

#include <uct/base/uct_md.h>
#include <ucs/sys/sock.h>
#ifdef __linux__
#include <linux/if.h>
#else
#include <net/if.h>
#endif

#define UCT_TCP_NAME "tcp"


/* How many events to wait for in epoll_wait */
#define UCT_TCP_MAX_EVENTS        16

/* If IFNAMSIZ is not found, set it to a large number */
#ifndef IFNAMSIZ
#define IFNAMSIZ 256
#endif

/* Forward declaration */
typedef struct uct_tcp_ep uct_tcp_ep_t;

typedef unsigned (*uct_tcp_ep_progress_t)(uct_tcp_ep_t *ep);


/**
 * TCP active message header
 */
typedef struct uct_tcp_am_hdr {
    uint8_t                       am_id;
    uint32_t                      length;
} UCS_S_PACKED uct_tcp_am_hdr_t;


/**
 * TCP endpoint communication context
 */
typedef struct uct_tcp_ep_ctx {
    void                          *buf;      /* Partial send/recv data */
    size_t                        length;    /* How much data in the buffer */
    size_t                        offset;    /* Next offset to send/recv */
    uct_tcp_ep_progress_t         progress;  /* Progress engine */
} uct_tcp_ep_ctx_t;


/**
 * TCP endpoint
 */
struct uct_tcp_ep {
    uct_base_ep_t                 super;
    int                           fd;          /* Socket file descriptor */
    uint32_t                      events;      /* Current notifications */
    uct_tcp_ep_ctx_t              tx;          /* TX resources */
    uct_tcp_ep_ctx_t              rx;          /* RX resources */
    ucs_sock_addr_t               peer_addr;   /* Remote iface addr */
    ucs_queue_head_t              pending_q;   /* Pending operations */
    ucs_list_link_t               list;
};


/**
 * TCP interface
 */
typedef struct uct_tcp_iface {
    uct_base_iface_t              super;             /* Parent class */
    int                           listen_fd;         /* Server socket */
    ucs_list_link_t               ep_list;           /* List of endpoints */
    char                          if_name[IFNAMSIZ]; /* Network interface name */
    int                           epfd;              /* Event poll set of sockets */
    size_t                        outstanding;       /* How much data in the EP send buffers */
    ucs_mpool_t                   tx_mpool;          /* TX memory pool */
    ucs_mpool_t                   rx_mpool;          /* RX memory pool */
    size_t                        am_buf_size;       /* AM buffer size */

    struct {
        struct sockaddr_in        ifaddr;            /* Network address */
        struct sockaddr_in        netmask;           /* Network address mask */
        size_t                    buf_size;          /* Maximal bcopy size */
        size_t                    short_size;        /* Maximal short size */
        int                       prefer_default;    /* Prefer default gateway */
        unsigned                  max_poll;          /* Number of events to poll per socket*/
    } config;

    struct {
        int                       nodelay;           /* TCP_NODELAY */
        int                       sndbuf;            /* SO_SNDBUF */
    } sockopt;
} uct_tcp_iface_t;


/**
 * TCP interface configuration
 */
typedef struct uct_tcp_iface_config {
    uct_iface_config_t            super;
    int                           prefer_default;
    unsigned                      max_poll;
    int                           sockopt_nodelay;
    size_t                        sockopt_sndbuf;
    uct_iface_mpool_config_t      tx_mpool;
    uct_iface_mpool_config_t      rx_mpool;
} uct_tcp_iface_config_t;


extern uct_md_component_t uct_tcp_md;
extern const char *uct_tcp_address_type_names[];

ucs_status_t uct_tcp_netif_caps(const char *if_name, double *latency_p,
                                double *bandwidth_p);

ucs_status_t uct_tcp_netif_inaddr(const char *if_name, struct sockaddr_in *ifaddr,
                                  struct sockaddr_in *netmask);

ucs_status_t uct_tcp_netif_is_default(const char *if_name, int *result_p);

int uct_tcp_sockaddr_cmp(const struct sockaddr *sa1,
                         const struct sockaddr *sa2);

ucs_status_t uct_tcp_send(int fd, const void *data, size_t *length_p);

ucs_status_t uct_tcp_recv(int fd, void *data, size_t *length_p);

ucs_status_t uct_tcp_send_blocking(int fd, const void *data, size_t length);

ucs_status_t uct_tcp_recv_blocking(int fd, void *data, size_t length);

ucs_status_t uct_tcp_iface_set_sockopt(uct_tcp_iface_t *iface, int fd);

ucs_status_t uct_tcp_ep_create(uct_tcp_iface_t *iface, int fd,
                               const struct sockaddr *dest_addr,
                               uct_tcp_ep_t **ep_p);

ucs_status_t uct_tcp_ep_create_connected(const uct_ep_params_t *params,
                                         uct_ep_h *ep_p);

void uct_tcp_ep_destroy(uct_ep_h tl_ep);

unsigned uct_tcp_ep_progress_tx(uct_tcp_ep_t *ep);

unsigned uct_tcp_ep_progress_rx(uct_tcp_ep_t *ep);

void uct_tcp_ep_mod_events(uct_tcp_ep_t *ep, uint32_t add, uint32_t remove);

ucs_status_t uct_tcp_ep_am_short(uct_ep_h uct_ep, uint8_t am_id, uint64_t header,
                                 const void *payload, unsigned length);

ssize_t uct_tcp_ep_am_bcopy(uct_ep_h uct_ep, uint8_t am_id,
                            uct_pack_callback_t pack_cb, void *arg,
                            unsigned flags);

ucs_status_t uct_tcp_ep_pending_add(uct_ep_h tl_ep, uct_pending_req_t *req,
                                    unsigned flags);

void uct_tcp_ep_pending_purge(uct_ep_h tl_ep, uct_pending_purge_callback_t cb,
                              void *arg);

ucs_status_t uct_tcp_ep_flush(uct_ep_h tl_ep, unsigned flags,
                              uct_completion_t *comp);

#endif
