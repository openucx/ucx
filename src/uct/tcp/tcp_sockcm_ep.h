/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "tcp_listener.h"


typedef enum uct_tcp_sockcm_ep_state {
    UCT_TCP_SOCKCM_EP_ON_SERVER     = UCS_BIT(0),  /* ep is on the server side */
    UCT_TCP_SOCKCM_EP_ON_CLIENT     = UCS_BIT(1),  /* ep is on the client side */
    UCT_TCP_SOCKCM_EP_CONNECTED     = UCS_BIT(2),  /* connect()/accept()
                                                      completed successfully */
    UCT_TCP_SOCKCM_EP_SENDING       = UCS_BIT(3),  /* ep is sending data */
    UCT_TCP_SOCKCM_EP_DATA_SENT     = UCS_BIT(4),  /* ep finished sending the data */
    UCT_TCP_SOCKCM_EP_RECEIVING     = UCS_BIT(5),  /* ep so receiving data */
    UCT_TCP_SOCKCM_EP_DATA_RECEIVED = UCS_BIT(6),  /* ep finished receviing the data */

    /* a connected ep is sending data */
    UCT_TCP_SOCKCM_EP_CONN_SENDING  =  UCT_TCP_SOCKCM_EP_CONNECTED |
                                       UCT_TCP_SOCKCM_EP_SENDING,

    /* a connected ep completed the data sending */
    UCT_TCP_SOCKCM_EP_CONN_SENT     =  UCT_TCP_SOCKCM_EP_CONNECTED |
                                       UCT_TCP_SOCKCM_EP_SENDING   |
                                       UCT_TCP_SOCKCM_EP_DATA_SENT,

    /* a connected ep is is receiving data */
    UCT_TCP_SOCKCM_EP_CONN_RECEIVING = UCT_TCP_SOCKCM_EP_CONNECTED |
                                       UCT_TCP_SOCKCM_EP_RECEIVING,

    /* a connected ep completed the data receiving */
    UCT_TCP_SOCKCM_EP_CONN_RECEIVED  = UCT_TCP_SOCKCM_EP_CONNECTED |
                                       UCT_TCP_SOCKCM_EP_RECEIVING |
                                       UCT_TCP_SOCKCM_EP_DATA_RECEIVED
} uct_tcp_sockcm_ep_state_t;


/**
 * TCP SOCKCM endpoint that is opened on a connection manager
 */
struct uct_tcp_sockcm_ep {
    uct_cm_base_ep_t   super;
    int                fd;        /* the fd of the socket on the ep */
    uint16_t           state;     /* ep state (uct_tcp_sockcm_ep_state_t) */
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

ucs_status_t uct_tcp_sockcm_ep_create(const uct_ep_params_t *params, uct_ep_h* ep_p);

ucs_status_t uct_tcp_sockcm_ep_disconnect(uct_ep_h ep, unsigned flags);

ucs_status_t uct_tcp_sockcm_ep_send_priv_data(uct_tcp_sockcm_ep_t *cep);

ucs_status_t uct_tcp_sockcm_ep_progress_send(uct_tcp_sockcm_ep_t *cep);

ucs_status_t uct_tcp_sockcm_ep_recv(uct_tcp_sockcm_ep_t *cep);

ucs_status_t uct_tcp_sockcm_ep_progress_recv(uct_tcp_sockcm_ep_t *cep);

size_t uct_tcp_sockcm_ep_get_priv_data_len(uct_tcp_sockcm_ep_t *cep);
