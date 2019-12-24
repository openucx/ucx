/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "tcp_listener.h"


typedef enum uct_tcp_sockcm_ep_state {
    UCT_TCP_SOCKCM_EP_ON_SERVER      = UCS_BIT(0),
    UCT_TCP_SOCKCM_EP_ON_CLIENT      = UCS_BIT(1),
    UCT_TCP_SOCKCM_EP_INIT           = UCS_BIT(2),
    UCT_TCP_SOCKCM_EP_CONNECTED      = UCS_BIT(3), /* connect()/accept()
                                                      completed successfully */
    UCT_TCP_SOCKCM_EP_SENDING        = UCS_BIT(4),
    UCT_TCP_SOCKCM_EP_DATA_SENT      = UCS_BIT(5),
    UCT_TCP_SOCKCM_EP_DATA_RECEIVING = UCS_BIT(6),
    UCT_TCP_SOCKCM_EP_DATA_RECVED    = UCS_BIT(7),
    UCT_TCP_SOCKCM_EP_DISCONNECTING  = UCS_BIT(8),  /* uct_ep_disconnect was
                                                       called on the ep.
                                                       this ep is not
                                                       necessarily disconnected
                                                       yet */
    UCT_TCP_SOCKCM_EP_FAILED         = UCS_BIT(9)
} uct_tcp_sockcm_ep_state_t;


/**
 * TCP sockcm endpoint communication context
 */
typedef struct uct_tcp_sockcm_ep_ctx {
    void                        *buf;           /* Data buffer to send */
    size_t                      total_length;   /* How much data to send */
    size_t                      offset;         /* Next offset to send */
} uct_tcp_sockcm_ep_ctx_t;


/**
 * TCP SOCKCM endpoint that is opened on a connection manager
 */
struct uct_tcp_sockcm_ep {
    uct_base_ep_t               super;
   int                          fd;
    void                        *user_data;    /* User data associated with the endpoint */
    uct_ep_disconnect_cb_t      disconnect_cb; /* Callback to handle the disconnection
                                                  of the remote peer */
    uint16_t                    state;
    uct_cm_sockaddr_wireup_cb_t wireup;
    uct_tcp_sockcm_ep_ctx_t     send;
};


UCS_CLASS_DECLARE_NEW_FUNC(uct_tcp_sockcm_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_tcp_sockcm_ep_t, uct_ep_t);

ucs_status_t uct_tcp_sockcm_ep_disconnect(uct_ep_h ep, unsigned flags);
