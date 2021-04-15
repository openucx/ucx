/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "tcp_listener.h"


typedef enum uct_tcp_sockcm_ep_state {
    UCT_TCP_SOCKCM_EP_ON_SERVER                   = UCS_BIT(0),  /* ep is on the server side */
    UCT_TCP_SOCKCM_EP_ON_CLIENT                   = UCS_BIT(1),  /* ep is on the client side */
    UCT_TCP_SOCKCM_EP_SERVER_CREATED              = UCS_BIT(2),  /* server's ep after call to uct_ep_create */
    UCT_TCP_SOCKCM_EP_PRIV_DATA_PACKED            = UCS_BIT(3),  /* ep packed its private data */
    UCT_TCP_SOCKCM_EP_HDR_RECEIVED                = UCS_BIT(4),  /* ep received the header of a new message */
    UCT_TCP_SOCKCM_EP_DATA_SENT                   = UCS_BIT(5),  /* ep finished sending the data */
    UCT_TCP_SOCKCM_EP_DATA_RECEIVED               = UCS_BIT(6),  /* ep finished receiving the data */
    UCT_TCP_SOCKCM_EP_CLIENT_CONNECTED_CB_INVOKED = UCS_BIT(7),  /* ep invoked the connect_cb on the client side */
    UCT_TCP_SOCKCM_EP_SERVER_NOTIFY_CB_INVOKED    = UCS_BIT(8),  /* ep invoked the notify_cb on the server side */
    UCT_TCP_SOCKCM_EP_CLIENT_NOTIFY_CALLED        = UCS_BIT(9),  /* ep on the client called notify API call */
    UCT_TCP_SOCKCM_EP_CLIENT_NOTIFY_SENT          = UCS_BIT(10), /* ep on the client sent the notify message to the server */
    UCT_TCP_SOCKCM_EP_DISCONNECTING               = UCS_BIT(11), /* @ref uct_ep_disconnect was called on the ep */
    UCT_TCP_SOCKCM_EP_DISCONNECTED                = UCS_BIT(12), /* ep is calling the upper layer disconnect callback */
    UCT_TCP_SOCKCM_EP_FAILED                      = UCS_BIT(13), /* ep is in error state due to an internal local error */
    UCT_TCP_SOCKCM_EP_CLIENT_GOT_REJECT           = UCS_BIT(14), /* ep on the client side received a reject from the server
                                                                    (debug flag) */
    UCT_TCP_SOCKCM_EP_PACK_CB_FAILED              = UCS_BIT(15), /* the upper layer's priv_pack_cb failed */
    UCT_TCP_SOCKCM_EP_SERVER_REJECT_CALLED        = UCS_BIT(16), /* ep on the server called reject API call */
    UCT_TCP_SOCKCM_EP_SERVER_REJECT_SENT          = UCS_BIT(17), /* ep on the server sent the reject message to the client */
    UCT_TCP_SOCKCM_EP_RESOLVE_CB_FAILED           = UCS_BIT(18), /* the upper layer's resolve_cb failed */
    UCT_TCP_SOCKCM_EP_RESOLVE_CB_INVOKED          = UCS_BIT(19), /* resolve_cb invoked */
    UCT_TCP_SOCKCM_EP_SERVER_CONN_REQ_CB_INVOKED  = UCS_BIT(20)  /* server ep was passed to a user via conn_req_cb */
} uct_tcp_sockcm_ep_state_t;


/**
 * TCP SOCKCM endpoint that is opened on a connection manager
 */
struct uct_tcp_sockcm_ep {
    uct_cm_base_ep_t   super;
    int                fd;        /* the fd of the socket on the ep */
    uint32_t           state;     /* ep state (uct_tcp_sockcm_ep_state_t) */
    uct_tcp_listener_t *listener; /* the listener the ep belongs to - used on the server side */
    ucs_list_link_t    list;      /* list item on the cm ep_list - used on the server side */
    struct {
        void           *buf;      /* Data buffer to send/recv */
        size_t         length;    /* How much data to send/recv */
        size_t         offset;    /* Next offset to send/recv */
    } comm_ctx;
};

UCS_CLASS_DECLARE(uct_tcp_sockcm_ep_t, const uct_ep_params_t *);
UCS_CLASS_DECLARE_NEW_FUNC(uct_tcp_sockcm_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_tcp_sockcm_ep_t, uct_ep_t);

static UCS_F_ALWAYS_INLINE
uct_tcp_sockcm_t *uct_tcp_sockcm_ep_get_cm(uct_tcp_sockcm_ep_t *cep)
{
    /* return the tcp sockcm connection manager this ep is using */
    return ucs_container_of(cep->super.super.super.iface, uct_tcp_sockcm_t,
                            super.iface);
}

void uct_tcp_sockcm_ep_close_fd(int *fd);

ucs_status_t uct_tcp_sockcm_ep_create(const uct_ep_params_t *params, uct_ep_h* ep_p);

ucs_status_t uct_tcp_sockcm_ep_connect(uct_ep_h ep,
                                       const uct_ep_connect_params_t *params);

ucs_status_t uct_tcp_sockcm_ep_disconnect(uct_ep_h ep, unsigned flags);

ucs_status_t uct_tcp_sockcm_ep_send(uct_tcp_sockcm_ep_t *cep);

ucs_status_t uct_tcp_sockcm_ep_progress_send(uct_tcp_sockcm_ep_t *cep);

ucs_status_t uct_tcp_sockcm_ep_recv(uct_tcp_sockcm_ep_t *cep);

ucs_status_t uct_tcp_sockcm_ep_set_sockopt(uct_tcp_sockcm_ep_t *ep);

ucs_status_t uct_tcp_sockcm_cm_ep_conn_notify(uct_ep_h ep);

const char *uct_tcp_sockcm_cm_ep_peer_addr_str(uct_tcp_sockcm_ep_t *cep,
                                               char *buf, size_t max);

void uct_tcp_sockcm_close_ep(uct_tcp_sockcm_ep_t *ep);

void uct_tcp_sockcm_ep_handle_event_status(uct_tcp_sockcm_ep_t *ep,
                                           ucs_status_t status,
                                           ucs_event_set_types_t events,
                                           const char *reason);
