/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_TCP_MD_H
#define UCT_TCP_MD_H

#include "tcp_base.h"

#include <uct/base/uct_md.h>
#include <uct/base/uct_iface.h>
#include <uct/base/uct_iov.inl>
#include <ucs/sys/sock.h>
#include <ucs/sys/string.h>
#include <ucs/datastruct/conn_match.h>
#include <ucs/datastruct/ptr_map.inl>
#include <ucs/algorithm/crc.h>
#include <ucs/sys/event_set.h>
#include <ucs/sys/iovec.h>

#include <net/if.h>

#define UCT_TCP_NAME                          "tcp"

#define UCT_TCP_CONFIG_PREFIX                 "TCP_"

/* Magic number that is used by TCP to identify its peers */
#define UCT_TCP_MAGIC_NUMBER                  0xCAFEBABE12345678lu

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

/* TX and RX caps */
#define UCT_TCP_EP_CTX_CAPS                  (UCT_TCP_EP_FLAG_CTX_TYPE_TX | \
                                              UCT_TCP_EP_FLAG_CTX_TYPE_RX)

/* Maximal value for connection sequence number */
#define UCT_TCP_CM_CONN_SN_MAX               UINT64_MAX


/**
 * TCP EP connection manager ID
 */
typedef union uct_tcp_ep_cm_id {
    ucs_conn_sn_t              conn_sn;        /* Connection sequence number, used by EPs
                                                * created with CONNECT_TO_IFACE method */
    ucs_ptr_map_key_t          ptr_map_key;    /* PTR map key, used by EPs created with
                                                * CONNECT_TO_EP method */
} uct_tcp_ep_cm_id_t;


/**
 * TCP EP flags
 */
enum {
    /* EP is connected to a peer to send data. This EP is managed
     * by a user and TCP mustn't free this EP even if connection
     * is broken. */
    UCT_TCP_EP_FLAG_CTX_TYPE_TX        = UCS_BIT(0),
    /* EP is connected to a peer to receive data. If only RX is set
     * on a given EP, it is hidden from a user (i.e. the user is unable
     * to do any operation on that EP) and TCP is responsible to
     * free memory allocating for this EP. */
    UCT_TCP_EP_FLAG_CTX_TYPE_RX        = UCS_BIT(1),
    /* Zcopy TX operation is in progress on a given EP. */
    UCT_TCP_EP_FLAG_ZCOPY_TX           = UCS_BIT(2),
    /* PUT RX operation is in progress on a given EP. */
    UCT_TCP_EP_FLAG_PUT_RX             = UCS_BIT(3),
    /* PUT TX operation is waiting for an ACK on a given EP. */
    UCT_TCP_EP_FLAG_PUT_TX_WAITING_ACK = UCS_BIT(4),
    /* PUT RX operation is waiting for resources to send an ACK
     * for received PUT operations on a given EP. */
    UCT_TCP_EP_FLAG_PUT_RX_SENDING_ACK = UCS_BIT(5),
    /* EP is on connection matching context. */
    UCT_TCP_EP_FLAG_ON_MATCH_CTX       = UCS_BIT(6),
    /* EP failed and a callback for handling error is scheduled. */
    UCT_TCP_EP_FLAG_FAILED             = UCS_BIT(7),
    /* EP is created to utilize CONNECT_TO_EP connection establishment
     * method. */
    UCT_TCP_EP_FLAG_CONNECT_TO_EP      = UCS_BIT(8),
    /* EP is on EP PTR map. */
    UCT_TCP_EP_FLAG_ON_PTR_MAP         = UCS_BIT(9)
};


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
    /* EP is receiving the magic number in order to verify a peer. EP is moved
     * to this state after accept() completed. */
    UCT_TCP_EP_CONN_STATE_RECV_MAGIC_NUMBER,
    /* EP is accepting connection from a peer, i.e. accept() returns socket fd
     * on which a connection was accepted, this EP was created using this socket
     * fd and the magic number was received and verified by EP and now it is
     * waiting for `UCT_TCP_CM_CONN_REQ` from a peer. */
    UCT_TCP_EP_CONN_STATE_ACCEPTING,
    /* EP is waiting for `UCT_TCP_CM_CONN_ACK` message from a peer after sending
     * `UCT_TCP_CM_CONN_REQ`.
     * All AM operations return `UCS_ERR_NO_RESOURCE` error to a caller. */
    UCT_TCP_EP_CONN_STATE_WAITING_ACK,
    /* EP is connected to a peer and they can communicate with each other. */
    UCT_TCP_EP_CONN_STATE_CONNECTED
} uct_tcp_ep_conn_state_t;

/* Forward declaration */
typedef struct uct_tcp_ep uct_tcp_ep_t;

typedef ucs_callback_t uct_tcp_ep_progress_t;


/**
 * TCP Connection Manager state
 */
typedef struct uct_tcp_cm_state {
    const char            *name;       /* CM state name */
    uct_tcp_ep_progress_t tx_progress; /* TX progress function */
    uct_tcp_ep_progress_t rx_progress; /* RX progress function */
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
    /* Connection acknowledgment + Connection request. The mesasge is sent
     * from a EP that accepts remote conenction when it was in
     * `UCT_TCP_EP_CONN_STATE_CONNECTING` state (i.e. original
     * `UCT_TCP_CM_CONN_REQ` wasn't sent yet) and want to have RX capability
     * on a peer's EP in order to send AM data. */
    UCT_TCP_CM_CONN_ACK_WITH_REQ      = (UCT_TCP_CM_CONN_REQ |
                                         UCT_TCP_CM_CONN_ACK)
} uct_tcp_cm_conn_event_t;


/**
 * TCP connection request packet flags
 */
enum {
    /* Inditicates whether both EPs of the connection has to use CONNECT_TO_EP
     * CONNECT_TO_EP of connection establishmnet */
    UCT_TCP_CM_CONN_REQ_PKT_FLAG_CONNECT_TO_EP = UCS_BIT(0)
};


/**
 * TCP connection request packet
 */
typedef struct uct_tcp_cm_conn_req_pkt {
    uct_tcp_cm_conn_event_t       event;      /* Connection event ID */
    uint8_t                       flags;      /* Packet flags */
    struct sockaddr_in            iface_addr; /* Socket address of UCT local iface */
    uct_tcp_ep_cm_id_t            cm_id;      /* EP connection mananger ID */
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
    size_t                        offset;         /* How much data was sent (TX) or was
                                                   * handled after receiving (RX) */
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
 * TCP endpoint address
 */
typedef struct uct_tcp_ep_addr {
    in_port_t                     iface_addr;     /* Interface address */
    ucs_ptr_map_key_t             ptr_map_key;    /* PTR map key, used by EPs created with
                                                   * CONNECT_TO_EP method */
} UCS_S_PACKED uct_tcp_ep_addr_t;


/**
 * TCP endpoint
 */
struct uct_tcp_ep {
    uct_base_ep_t                 super;
    uint8_t                       conn_retries;     /* Number of connection attempts done */
    uint8_t                       conn_state;       /* State of connection with peer */
    ucs_event_set_types_t         events;           /* Current notifications */
    uint16_t                      flags;            /* Endpoint flags */
    int                           fd;               /* Socket file descriptor */
    int                           stale_fd;         /* Old file descriptor which should be
                                                     * closed as soon as the EP is connected
                                                     * using the new fd */
    uct_tcp_ep_cm_id_t            cm_id;            /* EP connection mananger ID */
    uct_tcp_ep_ctx_t              tx;               /* TX resources */
    uct_tcp_ep_ctx_t              rx;               /* RX resources */
    struct sockaddr_in            peer_addr;        /* Remote iface addr */
    ucs_queue_head_t              pending_q;        /* Pending operations */
    ucs_queue_head_t              put_comp_q;       /* Flush completions waiting for
                                                     * outstanding PUTs acknowledgment */
    union {
        ucs_list_link_t           list;             /* List element to insert into TCP EP list */
        ucs_conn_match_elem_t     elem;             /* Connection matching element, used by EPs
                                                     * created with CONNECT_TO_IFACE method */
    };
};


/**
 * TCP interface
 */
typedef struct uct_tcp_iface {
    uct_base_iface_t              super;             /* Parent class */
    int                           listen_fd;         /* Server socket */
    ucs_conn_match_ctx_t          conn_match_ctx;    /* Connection matching context that contains EPs
                                                      * created with CONNECT_TO_IFACE method */
    ucs_ptr_map_t                 ep_ptr_map;        /* EP PTR map that contains EPs created
                                                      * with CONNECT_TO_EP method */
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
    ucs_range_spec_t              port_range;        /** Range of ports to use for bind() */

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
        int                       put_enable;        /* Enable PUT Zcopy operation support */
        int                       conn_nb;           /* Use non-blocking connect() */
        unsigned                  max_poll;          /* Number of events to poll per socket*/
        uint8_t                   max_conn_retries;  /* How many connection establishment attempts
                                                      * should be done if dropped connection was
                                                      * detected due to lack of system resources */
        unsigned                  syn_cnt;           /* Number of SYN retransmits that TCP should send
                                                      * before aborting the attempt to connect.
                                                      * It cannot exceed 255. */
        struct {
            ucs_time_t            idle;              /* The time the connection needs to remain
                                                      * idle before TCP starts sending keepalive
                                                      * probes (TCP_KEEPIDLE socket option) */
            unsigned              cnt;               /* The maximum number of keepalive probes TCP
                                                      * should send before dropping the connection
                                                      * (TCP_KEEPCNT socket option). */
            ucs_time_t            intvl;             /* The time between individual keepalive
                                                      * probes (TCP_KEEPINTVL socket option). */
        } keepalive;
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
    uct_iface_config_t             super;
    size_t                         tx_seg_size;
    size_t                         rx_seg_size;
    size_t                         max_iov;
    size_t                         sendv_thresh;
    int                            prefer_default;
    int                            put_enable;
    int                            conn_nb;
    unsigned                       max_poll;
    unsigned                       max_conn_retries;
    int                            sockopt_nodelay;
    uct_tcp_send_recv_buf_config_t sockopt;
    unsigned                       syn_cnt;
    uct_iface_mpool_config_t       tx_mpool;
    uct_iface_mpool_config_t       rx_mpool;
    ucs_range_spec_t               port_range;
    struct {
        ucs_time_t                 idle;
        unsigned                   cnt;
        ucs_time_t                 intvl;
    } keepalive;
} uct_tcp_iface_config_t;


extern uct_component_t uct_tcp_component;
extern const char *uct_tcp_address_type_names[];
extern const uct_tcp_cm_state_t uct_tcp_ep_cm_state[];
extern const ucs_conn_match_ops_t uct_tcp_cm_conn_match_ops;
extern const uct_tcp_ep_progress_t uct_tcp_ep_progress_rx_cb[];

ucs_status_t uct_tcp_netif_caps(const char *if_name, double *latency_p,
                                double *bandwidth_p);

ucs_status_t uct_tcp_netif_inaddr(const char *if_name, struct sockaddr_in *ifaddr,
                                  struct sockaddr_in *netmask);

ucs_status_t uct_tcp_netif_is_default(const char *if_name, int *result_p);

int uct_tcp_sockaddr_cmp(const struct sockaddr *sa1,
                         const struct sockaddr *sa2);

ucs_status_t uct_tcp_iface_set_sockopt(uct_tcp_iface_t *iface, int fd,
                                       int set_nb);

size_t uct_tcp_iface_get_max_iov(const uct_tcp_iface_t *iface);

size_t uct_tcp_iface_get_max_zcopy_header(const uct_tcp_iface_t *iface);

void uct_tcp_iface_add_ep(uct_tcp_ep_t *ep);

void uct_tcp_iface_remove_ep(uct_tcp_ep_t *ep);

int uct_tcp_cm_ep_accept_conn(uct_tcp_ep_t *ep);

int uct_tcp_iface_is_self_addr(uct_tcp_iface_t *iface,
                               const struct sockaddr_in *peer_addr);

ucs_status_t uct_tcp_ep_handle_io_err(uct_tcp_ep_t *ep, const char *op_str,
                                      ucs_status_t io_status);

ucs_status_t uct_tcp_ep_init(uct_tcp_iface_t *iface, int fd,
                             const struct sockaddr_in *dest_addr,
                             uct_tcp_ep_t **ep_p);

uint64_t uct_tcp_ep_get_cm_id(const uct_tcp_ep_t *ep);

ucs_status_t uct_tcp_ep_create(const uct_ep_params_t *params,
                               uct_ep_h *ep_p);

ucs_status_t uct_tcp_ep_get_address(uct_ep_h tl_ep, uct_ep_addr_t *ep_addr);

ucs_status_t uct_tcp_ep_connect_to_ep(uct_ep_h ep,
                                      const uct_device_addr_t *dev_addr,
                                      const uct_ep_addr_t *ep_addr);

const char *uct_tcp_ep_ctx_caps_str(uint8_t ep_ctx_caps, char *str_buffer);

void uct_tcp_ep_change_ctx_caps(uct_tcp_ep_t *ep, uint16_t new_caps);

void uct_tcp_ep_add_ctx_cap(uct_tcp_ep_t *ep, uint16_t cap);

void uct_tcp_ep_remove_ctx_cap(uct_tcp_ep_t *ep, uint16_t cap);

void uct_tcp_ep_move_ctx_cap(uct_tcp_ep_t *from_ep, uct_tcp_ep_t *to_ep,
                             uint16_t ctx_cap);

void uct_tcp_ep_destroy_internal(uct_ep_h tl_ep);

void uct_tcp_ep_destroy(uct_ep_h tl_ep);

void uct_tcp_ep_set_failed(uct_tcp_ep_t *ep);

void uct_tcp_ep_replace_ep(uct_tcp_ep_t *to_ep, uct_tcp_ep_t *from_ep);

int uct_tcp_ep_is_self(const uct_tcp_ep_t *ep);

uct_tcp_ep_t* uct_tcp_ep_ptr_map_retrieve(uct_tcp_iface_t *iface,
                                          ucs_ptr_map_key_t ptr_map_key);

void uct_tcp_ep_remove(uct_tcp_iface_t *iface, uct_tcp_ep_t *ep);

void uct_tcp_ep_add(uct_tcp_iface_t *iface, uct_tcp_ep_t *ep);

void uct_tcp_ep_mod_events(uct_tcp_ep_t *ep, ucs_event_set_types_t add,
                           ucs_event_set_types_t rem);

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

ucs_status_t uct_tcp_cm_send_event(uct_tcp_ep_t *ep,
                                   uct_tcp_cm_conn_event_t event,
                                   int log_error);

unsigned uct_tcp_cm_handle_conn_pkt(uct_tcp_ep_t **ep_p, void *pkt, uint32_t length);

unsigned uct_tcp_cm_conn_progress(void *arg);

uct_tcp_ep_conn_state_t
uct_tcp_cm_set_conn_state(uct_tcp_ep_t *ep,
                          uct_tcp_ep_conn_state_t new_conn_state);

void uct_tcp_cm_change_conn_state(uct_tcp_ep_t *ep,
                                  uct_tcp_ep_conn_state_t new_conn_state);

void uct_tcp_cm_ep_set_conn_sn(uct_tcp_ep_t *ep);

uct_tcp_ep_t *uct_tcp_cm_get_ep(uct_tcp_iface_t *iface,
                                const struct sockaddr_in *dest_address,
                                ucs_conn_sn_t conn_sn,
                                uint8_t with_ctx_cap);

uct_tcp_ep_t *uct_tcp_cm_get_conn_to_ep(uct_tcp_iface_t *iface,
                                        const struct sockaddr_in *dest_address,
                                        ucs_conn_sn_t conn_sn,
                                        uint8_t with_ctx_cap);

void uct_tcp_cm_insert_ep(uct_tcp_iface_t *iface, uct_tcp_ep_t *ep);

void uct_tcp_cm_remove_ep(uct_tcp_iface_t *iface, uct_tcp_ep_t *ep);

ucs_status_t uct_tcp_cm_handle_incoming_conn(uct_tcp_iface_t *iface,
                                             const struct sockaddr_in *peer_addr,
                                             int fd);

ucs_status_t uct_tcp_cm_conn_start(uct_tcp_ep_t *ep);

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
