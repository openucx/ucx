/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_TCP_MD_H
#define UCT_TCP_MD_H

#include <uct/base/uct_md.h>
#include <ucs/sys/sock.h>
#include <net/if.h>

#define UCT_TCP_NAME "tcp"

/* How many events to wait for in epoll_wait */
#define UCT_TCP_MAX_EVENTS        16


/**
 * TCP context type
 */
typedef enum uct_tcp_ep_ctx_type {
    UCT_TCP_EP_CTX_TYPE_TX,
    UCT_TCP_EP_CTX_TYPE_RX,
    UCT_TCP_EP_CTX_TYPE_LAST
} uct_tcp_ep_ctx_type_t;


/**
 * TCP endpoint connection state
 */
typedef enum uct_tcp_ep_conn_state {
    UCT_TCP_EP_CONN_STATE_CLOSED,
    UCT_TCP_EP_CONN_STATE_CONNECTING,
    UCT_TCP_EP_CONN_STATE_ACCEPTING,
    UCT_TCP_EP_CONN_STATE_WAITING_ACK,
    UCT_TCP_EP_CONN_STATE_CONNECTED
} uct_tcp_ep_conn_state_t;

/* Forward declaration */
typedef struct uct_tcp_ep uct_tcp_ep_t;

typedef unsigned (*uct_tcp_ep_progress_t)(uct_tcp_ep_t *ep);


/**
 * TCP Connection Manager state
 */
typedef struct uct_tcp_cm_state {
    const char            *name;                              /* CM state name */
    uct_tcp_ep_progress_t progress[UCT_TCP_EP_CTX_TYPE_LAST]; /* TX and RX progress functions */
} uct_tcp_cm_state_t;


/**
 * TCP Connection Manager event
 */
typedef enum uct_tcp_cm_conn_event {
    UCT_TCP_CM_CONN_REQ,
    UCT_TCP_CM_CONN_ACK
} uct_tcp_cm_conn_event_t;


/**
 * TCP connection request packet
 */
typedef struct uct_tcp_cm_conn_req_pkt {
    uct_tcp_cm_conn_event_t       event;
    struct sockaddr_in            iface_addr;
} UCS_S_PACKED uct_tcp_cm_conn_req_pkt_t;


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
} uct_tcp_ep_ctx_t;


/**
 * TCP endpoint
 */
struct uct_tcp_ep {
    uct_base_ep_t                 super;
    uint8_t                       ctx_caps;    /* Which contexts are supported */
    int                           fd;          /* Socket file descriptor */
    uct_tcp_ep_conn_state_t       conn_state;  /* State of connection with peer */
    uint32_t                      events;      /* Current notifications */
    uct_tcp_ep_ctx_t              tx;          /* TX resources */
    uct_tcp_ep_ctx_t              rx;          /* RX resources */
    struct sockaddr_in            peer_addr;   /* Remote iface addr */
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
    ucs_mpool_t                   tx_mpool;          /* TX memory pool */
    ucs_mpool_t                   rx_mpool;          /* RX memory pool */
    size_t                        am_buf_size;       /* AM buffer size */
    size_t                        outstanding;       /* How much data in the EP send buffers
                                                      * + how many non-blocking connections
                                                      * are in progress */

    struct {
        struct sockaddr_in        ifaddr;            /* Network address */
        struct sockaddr_in        netmask;           /* Network address mask */
        int                       prefer_default;    /* Prefer default gateway */
        unsigned                  max_poll;          /* Number of events to poll per socket*/
    } config;

    struct {
        int                       nodelay;           /* TCP_NODELAY */
        int                       sndbuf;            /* SO_SNDBUF */
        int                       rcvbuf;            /* SO_RCVBUF */
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
    size_t                        sockopt_rcvbuf;
    uct_iface_mpool_config_t      tx_mpool;
    uct_iface_mpool_config_t      rx_mpool;
} uct_tcp_iface_config_t;


extern uct_md_component_t uct_tcp_md;
extern const char *uct_tcp_address_type_names[];
extern const uct_tcp_cm_state_t uct_tcp_ep_cm_state[];

ucs_status_t uct_tcp_netif_caps(const char *if_name, double *latency_p,
                                double *bandwidth_p);

ucs_status_t uct_tcp_netif_inaddr(const char *if_name, struct sockaddr_in *ifaddr,
                                  struct sockaddr_in *netmask);

ucs_status_t uct_tcp_netif_is_default(const char *if_name, int *result_p);

int uct_tcp_sockaddr_cmp(const struct sockaddr *sa1,
                         const struct sockaddr *sa2);

ucs_status_t uct_tcp_iface_set_sockopt(uct_tcp_iface_t *iface, int fd);

void uct_tcp_iface_outstanding_inc(uct_tcp_iface_t *iface);

void uct_tcp_iface_outstanding_dec(uct_tcp_iface_t *iface);

ucs_status_t uct_tcp_ep_init(uct_tcp_iface_t *iface, int fd,
                             const struct sockaddr_in *dest_addr,
                             uct_tcp_ep_t **ep_p);

ucs_status_t uct_tcp_ep_create(const uct_ep_params_t *params,
                               uct_ep_h *ep_p);

void uct_tcp_ep_destroy(uct_ep_h tl_ep);

void uct_tcp_ep_set_failed(uct_tcp_ep_t *ep);

void uct_tcp_ep_remove(uct_tcp_iface_t *iface, uct_tcp_ep_t *ep);

void uct_tcp_ep_add(uct_tcp_iface_t *iface, uct_tcp_ep_t *ep);

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

unsigned uct_tcp_cm_conn_progress(uct_tcp_ep_t *ep);

unsigned uct_tcp_cm_conn_ack_rx_progress(uct_tcp_ep_t *ep);

unsigned uct_tcp_cm_conn_req_rx_progress(uct_tcp_ep_t *ep);

void uct_tcp_cm_change_conn_state(uct_tcp_ep_t *ep,
                                  uct_tcp_ep_conn_state_t new_conn_state);

ucs_status_t uct_tcp_cm_handle_incoming_conn(uct_tcp_iface_t *iface,
                                             const struct sockaddr_in *peer_addr,
                                             int fd);

ucs_status_t uct_tcp_cm_conn_start(uct_tcp_ep_t *ep);

static inline unsigned
uct_tcp_ep_progress(uct_tcp_ep_t *ep, uct_tcp_ep_ctx_type_t ctx_type)
{
    return uct_tcp_ep_cm_state[ep->conn_state].progress[ctx_type](ep);
}


#endif
