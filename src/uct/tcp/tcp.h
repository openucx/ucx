/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_TCP_MD_H
#define UCT_TCP_MD_H

#include <uct/base/uct_md.h>
#include <ucs/sys/sock.h>
#include <ucs/sys/string.h>
#include <ucs/datastruct/khash.h>
#include <ucs/algorithm/crc.h>
#include <ucs/sys/event_set.h>

#include <net/if.h>

#define UCT_TCP_NAME                "tcp"

/* Maximum number of events to wait from event set */
#define UCT_TCP_MAX_EVENTS          16

/* How long should be string to keep [%s:%s] string
 * where %s value can be -/Tx/Rx */
#define UCT_TCP_EP_CTX_CAPS_STR_MAX 8

/**
 * TCP context type
 */
typedef enum uct_tcp_ep_ctx_type {
    UCT_TCP_EP_CTX_TYPE_TX,
    UCT_TCP_EP_CTX_TYPE_RX
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

static inline int uct_tcp_khash_sockaddr_in_equal(struct sockaddr_in sa1,
                                                  struct sockaddr_in sa2)
{
    ucs_status_t status;
    int cmp;

    cmp = ucs_sockaddr_cmp((const struct sockaddr*)&sa1,
                           (const struct sockaddr*)&sa2,
                           &status);
    ucs_assert(status == UCS_OK);
    return !cmp;
}

static inline uint32_t uct_tcp_khash_sockaddr_in_hash(struct sockaddr_in sa)
{
    ucs_status_t UCS_V_UNUSED status;
    size_t addr_size;

    status = ucs_sockaddr_sizeof((const struct sockaddr*)&sa,
                                 &addr_size);
    ucs_assert(status == UCS_OK);
    return ucs_crc32(0, (const void *)&sa, addr_size);
}

KHASH_INIT(uct_tcp_cm_eps, struct sockaddr_in, ucs_list_link_t*,
           1, uct_tcp_khash_sockaddr_in_hash, uct_tcp_khash_sockaddr_in_equal);


/**
 * TCP Connection Manager state
 */
typedef struct uct_tcp_cm_state {
    const char            *name;       /* CM state name */
    uct_tcp_ep_progress_t tx_progress; /* TX progress function */
} uct_tcp_cm_state_t;


/**
 * TCP Connection Manager event
 */
typedef enum uct_tcp_cm_conn_event {
    UCT_TCP_CM_CONN_REQ          = UCS_BIT(0),
    UCT_TCP_CM_CONN_ACK          = UCS_BIT(1),
    UCT_TCP_CM_CONN_ACK_WITH_REQ = UCT_TCP_CM_CONN_REQ | UCT_TCP_CM_CONN_ACK,
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
    int                           events;      /* Current notifications */
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
    khash_t(uct_tcp_cm_eps)       ep_cm_map;         /* Map of endpoints that don't
                                                      * have one of the context cap */
    ucs_list_link_t               ep_list;           /* List of endpoints */
    char                          if_name[IFNAMSIZ]; /* Network interface name */
    ucs_sys_event_set_t           *event_set;        /* Event set identifier */
    ucs_mpool_t                   tx_mpool;          /* TX memory pool */
    ucs_mpool_t                   rx_mpool;          /* RX memory pool */
    size_t                        seg_size;          /* AM buffer size */
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
        size_t                    sndbuf;            /* SO_SNDBUF */
        size_t                    rcvbuf;            /* SO_RCVBUF */
    } sockopt;
} uct_tcp_iface_t;


/**
 * TCP interface configuration
 */
typedef struct uct_tcp_iface_config {
    uct_iface_config_t            super;
    size_t                        seg_size;
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

void uct_tcp_iface_add_ep(uct_tcp_ep_t *ep);

void uct_tcp_iface_remove_ep(uct_tcp_ep_t *ep);

ucs_status_t uct_tcp_ep_init(uct_tcp_iface_t *iface, int fd,
                             const struct sockaddr_in *dest_addr,
                             uct_tcp_ep_t **ep_p);

ucs_status_t uct_tcp_ep_create(const uct_ep_params_t *params,
                               uct_ep_h *ep_p);

const char *uct_tcp_ep_ctx_caps_str(uint8_t ep_ctx_caps, char *str_buffer);

void uct_tcp_ep_change_ctx_caps(uct_tcp_ep_t *ep, uint8_t new_caps);

ucs_status_t uct_tcp_ep_add_ctx_cap(uct_tcp_ep_t *ep,
                                    uct_tcp_ep_ctx_type_t cap);

ucs_status_t uct_tcp_ep_remove_ctx_cap(uct_tcp_ep_t *ep,
                                       uct_tcp_ep_ctx_type_t cap);

ucs_status_t uct_tcp_ep_move_ctx_cap(uct_tcp_ep_t *from_ep, uct_tcp_ep_t *to_ep,
                                     uct_tcp_ep_ctx_type_t ctx_cap);

void uct_tcp_ep_destroy_internal(uct_ep_h tl_ep);

void uct_tcp_ep_destroy(uct_ep_h tl_ep);

void uct_tcp_ep_set_failed(uct_tcp_ep_t *ep);

unsigned uct_tcp_ep_is_self(const uct_tcp_ep_t *ep);

void uct_tcp_ep_remove(uct_tcp_iface_t *iface, uct_tcp_ep_t *ep);

void uct_tcp_ep_add(uct_tcp_iface_t *iface, uct_tcp_ep_t *ep);

unsigned uct_tcp_ep_progress_rx(uct_tcp_ep_t *ep);

void uct_tcp_ep_mod_events(uct_tcp_ep_t *ep, int add, int remove);

void uct_tcp_ep_pending_queue_dispatch(uct_tcp_ep_t *ep);

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

ucs_status_t uct_tcp_cm_send_event(uct_tcp_ep_t *ep, uct_tcp_cm_conn_event_t event);

unsigned uct_tcp_cm_handle_conn_pkt(uct_tcp_ep_t **ep, void *pkt, uint32_t length);

unsigned uct_tcp_cm_conn_progress(uct_tcp_ep_t *ep);

uct_tcp_ep_conn_state_t
uct_tcp_cm_set_conn_state(uct_tcp_ep_t *ep,
                          uct_tcp_ep_conn_state_t new_conn_state);

void uct_tcp_cm_change_conn_state(uct_tcp_ep_t *ep,
                                  uct_tcp_ep_conn_state_t new_conn_state);

ucs_status_t uct_tcp_cm_add_ep(uct_tcp_iface_t *iface, uct_tcp_ep_t *ep);

void uct_tcp_cm_remove_ep(uct_tcp_iface_t *iface, uct_tcp_ep_t *ep);

uct_tcp_ep_t *uct_tcp_cm_search_ep(uct_tcp_iface_t *iface,
                                   const struct sockaddr_in *peer_addr,
                                   uct_tcp_ep_ctx_type_t with_ctx_type);

void uct_tcp_cm_purge_ep(uct_tcp_ep_t *ep);

ucs_status_t uct_tcp_cm_handle_incoming_conn(uct_tcp_iface_t *iface,
                                             const struct sockaddr_in *peer_addr,
                                             int fd);

ucs_status_t uct_tcp_cm_conn_start(uct_tcp_ep_t *ep);

static inline unsigned uct_tcp_ep_progress_tx(uct_tcp_ep_t *ep)
{
    return uct_tcp_ep_cm_state[ep->conn_state].tx_progress(ep);
}


#endif
