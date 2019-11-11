/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_TCP_MD_H
#define UCT_TCP_MD_H

#include <uct/base/uct_md.h>
#include <uct/base/uct_iface.h>
#include <ucs/sys/sock.h>
#include <ucs/sys/string.h>
#include <ucs/datastruct/khash.h>
#include <ucs/algorithm/crc.h>
#include <ucs/sys/event_set.h>
#include <ucs/sys/iovec.h>

#include <net/if.h>

#define UCT_TCP_NAME                          "tcp"

#define UCT_TCP_CONFIG_PREFIX                 "TCP_"

/* Maximum number of events to wait on event set */
#define UCT_TCP_MAX_EVENTS                    16

/* How long should be string to keep [%s:%s] string
 * where %s value can be -/Tx/Rx */
#define UCT_TCP_EP_CTX_CAPS_STR_MAX           8

/* How many IOVs are needed to keep AM/PUT Zcopy service data
 * (TCP protocol and user's AM (or PUT) headers) */
#define UCT_TCP_EP_ZCOPY_SERVICE_IOV_COUNT    2

/* How many IOVs are needed to do AM Short
 * (TCP protocol and user's AM headers, payload) */
#define UCT_TCP_EP_AM_SHORTV_IOV_COUNT        3

/* Maximum size of a data that can be sent by PUT Zcopy
 * operation */
#define UCT_TCP_EP_PUT_ZCOPY_MAX              SIZE_MAX

/* Length of a data that is used by PUT protocol */
#define UCT_TCP_EP_PUT_SERVICE_LENGTH        (sizeof(uct_tcp_am_hdr_t) + \
                                              sizeof(uct_tcp_ep_put_req_hdr_t))

#define UCT_TCP_CONFIG_MAX_CONN_RETRIES      "MAX_CONN_RETRIES"


/**
 * TCP context type
 */
typedef enum uct_tcp_ep_ctx_type {
    /* EP is connected to a peer to send data. This EP is managed
     * by a user and TCP mustn't free this EP even if connection
     * is broken. */
    UCT_TCP_EP_CTX_TYPE_TX,
    /* EP is connected to a peer to receive data. If only RX is set
     * on a given EP, it is hidden from a user (i.e. the user is unable
     * to do any operation on that EP) and TCP is responsible to
     * free memory allocating for this EP. */
    UCT_TCP_EP_CTX_TYPE_RX,

    /* Additional flags that controls EP behavior: */
    /* - Zcopy TX operation is in progress on a given EP. */
    UCT_TCP_EP_CTX_TYPE_ZCOPY_TX,
    /* - PUT RX operation is in progress on a given EP. */
    UCT_TCP_EP_CTX_TYPE_PUT_RX,
    /* - PUT TX operation is waiting for an ACK on a given EP */
    UCT_TCP_EP_CTX_TYPE_PUT_TX_WAITING_ACK,
    /* - PUT RX operation is waiting for resources to send an ACK
     *   for received PUT operations on a given EP */
    UCT_TCP_EP_CTX_TYPE_PUT_RX_SENDING_ACK
} uct_tcp_ep_ctx_type_t;


/**
 * TCP endpoint connection state
 */
typedef enum uct_tcp_ep_conn_state {
    /* EP is unable to communicate with a peer's EP - connections establishment
     * was unsuccessful or detected hangup during communications. */
    UCT_TCP_EP_CONN_STATE_CLOSED,
    /* EP is connecting to a peer's EP, i.e. connect() was called on non-blocking
     * socket and returned this call returned that an operation is in progress.
     * After it is done, it sends `UCT_TCP_CM_CONN_REQ` to the peer.
     * All AM operations return `UCS_ERR_NO_RESOURCE` error to a caller. */
    UCT_TCP_EP_CONN_STATE_CONNECTING,
    /* EP is accepting connection from a peer, i.e. accept() returns socket fd
     * on which a connection was accepted, this EP was created using this socket
     * and now it is waiting for `UCT_TCP_CM_CONN_REQ` from a peer. */
    UCT_TCP_EP_CONN_STATE_ACCEPTING,
    /* EP is waiting for `UCT_TCP_CM_CONN_ACK` message from a peer after sending
     * `UCT_TCP_CM_CONN_REQ`.
     * All AM operations return `UCS_ERR_NO_RESOURCE` error to a caller. */
    UCT_TCP_EP_CONN_STATE_WAITING_ACK,
    /* EP is waiting for a connection and `UCT_TCP_CM_CONN_REQ` message from
     * a peer after simultaneous connection resolution between them. This EP
     * is a "winner" of the resolution, but no RX capability on this PR (i.e.
     * no `UCT_TCP_CM_CONN_REQ` message was received from the peer). EP is moved
     * to `UCT_TCP_EP_CONN_STATE_CONNECTED` state upon receiving this message.
     * All AM operations return `UCS_ERR_NO_RESOURCE` error to a caller. */
    UCT_TCP_EP_CONN_STATE_WAITING_REQ,
    /* EP is connected to a peer and them can comunicate with each other. */
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
    /* Connection request from a EP that has TX capability to a EP that
     * has to be able to receive AM data (i.e. has to have RX capability). */
    UCT_TCP_CM_CONN_REQ               = UCS_BIT(0),
    /* Connection acknowledgment from a EP that accepts a conenction from
     * initiator of a connection request. */
    UCT_TCP_CM_CONN_ACK               = UCS_BIT(1),
    /* Request for waiting of a connection request.
     * The mesage is not sent separately (only along with a connection
     * acknowledgment.) */
    UCT_TCP_CM_CONN_WAIT_REQ          = UCS_BIT(2),
    /* Connection acknowledgment + Connection request. The mesasge is sent
     * from a EP that accepts remote conenction when it was in
     * `UCT_TCP_EP_CONN_STATE_CONNECTING` state (i.e. original
     * `UCT_TCP_CM_CONN_REQ` wasn't sent yet) and want to have RX capability
     * on a peer's EP in order to send AM data. */
    UCT_TCP_CM_CONN_ACK_WITH_REQ      = (UCT_TCP_CM_CONN_REQ |
                                         UCT_TCP_CM_CONN_ACK),
    /* Connection acknowledgment + Request for waiting of a connection request.
     * The message is sent from a EP that accepts remote conenction when it was
     * in `UCT_TCP_EP_CONN_STATE_WAITING_ACK` state (i.e. original
     * `UCT_TCP_CM_CONN_REQ` was sent) and want to have RX capability on a
     * peer's EP in order to send AM data. */
    UCT_TCP_CM_CONN_ACK_WITH_WAIT_REQ = (UCT_TCP_CM_CONN_WAIT_REQ |
                                         UCT_TCP_CM_CONN_ACK)
} uct_tcp_cm_conn_event_t;


/**
 * TCP connection request packet
 */
typedef struct uct_tcp_cm_conn_req_pkt {
    uct_tcp_cm_conn_event_t       event;      /* Connection event ID */
    struct sockaddr_in            iface_addr; /* Socket address of UCT local iface */
} UCS_S_PACKED uct_tcp_cm_conn_req_pkt_t;


/**
 * TCP active message header
 */
typedef struct uct_tcp_am_hdr {
    uint8_t                       am_id;      /* UCT AM ID of an AM operation */
    uint32_t                      length;     /* Length of data sent in an AM operation */
} UCS_S_PACKED uct_tcp_am_hdr_t;


/**
 * AM IDs reserved for TCP protocols
 */
typedef enum uct_tcp_ep_am_id {
    /* AM ID reserved for TCP internal Connection Manager messages */
    UCT_TCP_EP_CM_AM_ID      = UCT_AM_ID_MAX,
    /* AM ID reserved for TCP internal PUT REQ message */
    UCT_TCP_EP_PUT_REQ_AM_ID = UCT_AM_ID_MAX + 1,
    /* AM ID reserved for TCP internal PUT ACK message */
    UCT_TCP_EP_PUT_ACK_AM_ID = UCT_AM_ID_MAX + 2
} uct_tcp_ep_am_id_t;


/**
 * TCP PUT request header
 */
typedef struct uct_tcp_ep_put_req_hdr {
    uint64_t                      addr;        /* Address of a remote memory buffer */
    size_t                        length;      /* Length of a remote memory buffer */
    uint32_t                      sn;          /* Sequence number of the current PUT operation */
} UCS_S_PACKED uct_tcp_ep_put_req_hdr_t;


/**
 * TCP PUT acknowledge header
 */
typedef struct uct_tcp_ep_put_ack_hdr {
    uint32_t                      sn;          /* Sequence number of the last acked PUT operation */
} UCS_S_PACKED uct_tcp_ep_put_ack_hdr_t;


/**
 * TCP PUT completion
 */
typedef struct uct_tcp_ep_put_completion {
    uct_completion_t              *comp;           /* User's completion passed to
                                                    * uct_ep_flush */
    uint32_t                      wait_put_sn;     /* Sequence number of the last unacked
                                                    * PUT operations that was in-progress
                                                    * when uct_ep_flush was called */
    ucs_queue_elem_t              elem;            /* Element to insert completion into
                                                    * TCP EP PUT operation pending queue */
} uct_tcp_ep_put_completion_t;


/**
 * TCP endpoint communication context
 */
typedef struct uct_tcp_ep_ctx {
    uint32_t                      put_sn;         /* Sequence number of last sent
                                                   * or received PUT operation */
    void                          *buf;           /* Partial send/recv data */
    size_t                        length;         /* How much data in the buffer */
    size_t                        offset;         /* Next offset to send/recv */
} uct_tcp_ep_ctx_t;


/**
 * TCP AM/PUT Zcopy communication context mapped to
 * buffer from TCP EP context
 */
typedef struct uct_tcp_ep_zcopy_tx {
    uct_tcp_am_hdr_t              super;     /* UCT TCP AM header */
    uct_completion_t              *comp;     /* Local UCT completion object */
    size_t                        iov_index; /* Current IOV index */
    size_t                        iov_cnt;   /* Number of IOVs that should be sent */
    struct iovec                  iov[0];    /* IOVs that should be sent */
} uct_tcp_ep_zcopy_tx_t;


/**
 * TCP endpoint
 */
struct uct_tcp_ep {
    uct_base_ep_t                 super;
    uint8_t                       ctx_caps;         /* Which contexts are supported */
    int                           fd;               /* Socket file descriptor */
    uct_tcp_ep_conn_state_t       conn_state;       /* State of connection with peer */
    unsigned                      conn_retries;     /* Number of connection attempts done */
    int                           events;           /* Current notifications */
    uct_tcp_ep_ctx_t              tx;               /* TX resources */
    uct_tcp_ep_ctx_t              rx;               /* RX resources */
    struct sockaddr_in            peer_addr;        /* Remote iface addr */
    ucs_queue_head_t              pending_q;        /* Pending operations */
    ucs_queue_head_t              put_comp_q;       /* Flush completions waiting for
                                                     * outstanding PUTs acknowledgment */
    ucs_list_link_t               list;             /* List element to insert into TCP EP list */
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
    size_t                        outstanding;       /* How much data in the EP send buffers
                                                      * + how many non-blocking connections
                                                      * are in progress + how many EPs are
                                                      * waiting for PUT Zcopy operation ACKs
                                                      * (0/1 for each EP) */

    struct {
        size_t                    tx_seg_size;       /* TX AM buffer size */
        size_t                    rx_seg_size;       /* RX AM buffer size */
        size_t                    sendv_thresh;      /* Minimum size of user's payload from which
                                                      * non-blocking vector send should be used */
        struct {
            size_t                max_iov;           /* Maximum supported IOVs limited by
                                                      * user configuration and service buffers
                                                      * (TCP protocol and user's AM headers) */
            size_t                max_hdr;           /* Maximum supported AM Zcopy header */
            size_t                hdr_offset;        /* Offset in TX buffer to empty space that
                                                      * can be used for AM Zcopy header */
        } zcopy;
        struct sockaddr_in        ifaddr;            /* Network address */
        struct sockaddr_in        netmask;           /* Network address mask */
        int                       prefer_default;    /* Prefer default gateway */
        int                       conn_nb;           /* Use non-blocking connect() */
        unsigned                  max_poll;          /* Number of events to poll per socket*/
        unsigned                  max_conn_retries;  /* How many connection establishment attmepts
                                                      * should be done if dropped connection was
                                                      * detected due to lack of system resources */
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
    size_t                        tx_seg_size;
    size_t                        rx_seg_size;
    size_t                        max_iov;
    size_t                        sendv_thresh;
    int                           prefer_default;
    int                           conn_nb;
    unsigned                      max_poll;
    unsigned                      max_conn_retries;
    int                           sockopt_nodelay;
    size_t                        sockopt_sndbuf;
    size_t                        sockopt_rcvbuf;
    uct_iface_mpool_config_t      tx_mpool;
    uct_iface_mpool_config_t      rx_mpool;
} uct_tcp_iface_config_t;


extern uct_component_t uct_tcp_component;
extern const char *uct_tcp_address_type_names[];
extern const uct_tcp_cm_state_t uct_tcp_ep_cm_state[];
extern const uct_tcp_ep_progress_t uct_tcp_ep_progress_rx_cb[];

ucs_status_t uct_tcp_netif_caps(const char *if_name, double *latency_p,
                                double *bandwidth_p);

ucs_status_t uct_tcp_netif_inaddr(const char *if_name, struct sockaddr_in *ifaddr,
                                  struct sockaddr_in *netmask);

ucs_status_t uct_tcp_netif_is_default(const char *if_name, int *result_p);

int uct_tcp_sockaddr_cmp(const struct sockaddr *sa1,
                         const struct sockaddr *sa2);

ucs_status_t uct_tcp_iface_set_sockopt(uct_tcp_iface_t *iface, int fd);

size_t uct_tcp_iface_get_max_iov(const uct_tcp_iface_t *iface);

size_t uct_tcp_iface_get_max_zcopy_header(const uct_tcp_iface_t *iface);

void uct_tcp_iface_add_ep(uct_tcp_ep_t *ep);

void uct_tcp_iface_remove_ep(uct_tcp_ep_t *ep);

ucs_status_t uct_tcp_ep_handle_dropped_connect(uct_tcp_ep_t *ep, int io_errno);

unsigned uct_tcp_ep_progress_am_rx(uct_tcp_ep_t *ep);

unsigned uct_tcp_ep_progress_put_rx(uct_tcp_ep_t *ep);

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

void uct_tcp_ep_mod_events(uct_tcp_ep_t *ep, int add, int remove);

void uct_tcp_ep_pending_queue_dispatch(uct_tcp_ep_t *ep);

ucs_status_t uct_tcp_ep_am_short(uct_ep_h uct_ep, uint8_t am_id, uint64_t header,
                                 const void *payload, unsigned length);

ssize_t uct_tcp_ep_am_bcopy(uct_ep_h uct_ep, uint8_t am_id,
                            uct_pack_callback_t pack_cb, void *arg,
                            unsigned flags);

ucs_status_t uct_tcp_ep_am_zcopy(uct_ep_h uct_ep, uint8_t am_id, const void *header,
                                 unsigned header_length, const uct_iov_t *iov,
                                 size_t iovcnt, unsigned flags,
                                 uct_completion_t *comp);

ucs_status_t uct_tcp_ep_put_zcopy(uct_ep_h uct_ep, const uct_iov_t *iov,
                                  size_t iovcnt, uint64_t remote_addr,
                                  uct_rkey_t rkey, uct_completion_t *comp);

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

static inline unsigned uct_tcp_ep_progress_rx(uct_tcp_ep_t *ep)
{
    if (!(ep->ctx_caps & UCS_BIT(UCT_TCP_EP_CTX_TYPE_PUT_RX))) {
        return uct_tcp_ep_progress_am_rx(ep);
    } else {
        return uct_tcp_ep_progress_put_rx(ep);
    }
}

static inline void uct_tcp_iface_outstanding_inc(uct_tcp_iface_t *iface)
{
    iface->outstanding++;
}

static inline void uct_tcp_iface_outstanding_dec(uct_tcp_iface_t *iface)
{
    ucs_assert(iface->outstanding > 0);
    iface->outstanding--;
}

/**
 * Query for active network devices under /sys/class/net, as determined by
 * ucs_netif_is_active(). 'md' parameter is not used, and is added for
 * compatibility with uct_tl_t::query_devices definition.
 */
ucs_status_t uct_tcp_query_devices(uct_md_h md,
                                   uct_tl_device_resource_t **devices_p,
                                   unsigned *num_devices_p);

#endif
