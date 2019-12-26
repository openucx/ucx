/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "tcp_listener.h"


typedef enum uct_tcp_sockcm_ep_state {
    UCT_TCP_SOCKCM_EP_ON_SERVER      = UCS_BIT(0), /* ep is on the server side */
    UCT_TCP_SOCKCM_EP_ON_CLIENT      = UCS_BIT(1), /* ep is on the client side */
    UCT_TCP_SOCKCM_EP_CONNECTED      = UCS_BIT(2), /* connect()/accept()
                                                      completed successfully */
    UCT_TCP_SOCKCM_EP_SENDING        = UCS_BIT(3), /* ep started sending data */
    UCT_TCP_SOCKCM_EP_DATA_SENT      = UCS_BIT(4), /* ep completed sending the data */
    UCT_TCP_SOCKCM_EP_DATA_RECEIVING = UCS_BIT(5), /* ep started receiving data */
    UCT_TCP_SOCKCM_EP_DATA_RECVED    = UCS_BIT(6), /* ep completed receiving the data */
    UCT_TCP_SOCKCM_EP_DISCONNECTING  = UCS_BIT(7), /* uct_ep_disconnect was
                                                      called on the ep.
                                                      this ep is not
                                                      necessarily disconnected
                                                      yet */
    UCT_TCP_SOCKCM_EP_FAILED         = UCS_BIT(8)  /* ep is in error state */
} uct_tcp_sockcm_ep_state_t;


/**
 * TCP SOCKCM endpoint that is opened on a connection manager
 */
struct uct_tcp_sockcm_ep {
    uct_cm_base_ep_t super;
    int              fd;
    uint16_t         state;
    struct {
        void         *buf;           /* Data buffer to send/recv */
        size_t       total_length;   /* How much data to send/recv */
        size_t       offset;         /* Next offset to send/recv */
    } comm_ctx;
};


UCS_CLASS_DECLARE_NEW_FUNC(uct_tcp_sockcm_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_tcp_sockcm_ep_t, uct_ep_t);

ucs_status_t uct_tcp_sockcm_ep_disconnect(uct_ep_h ep, unsigned flags);
