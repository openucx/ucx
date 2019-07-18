/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "rdmacm_listener.h"

/**
 * RDMACM endpoint that is opened on a connection manager
 */
typedef struct uct_rdmacm_cm_ep {
    uct_base_ep_t          super;
    uct_rdmacm_cm_t        *cm; /* Connection manager this ep is using */
    struct rdma_cm_id      *id; /* The rdmacm id that is created per this ep */
    struct ibv_cq          *cq; /* Dummy cq used for creating a dummy qp */
    struct ibv_qp          *qp; /* Dummy qp used for generating a unique qp_num */
    void                   *user_data;    /* User data associated with the endpoint */
    uct_ep_disconnect_cb_t disconnect_cb; /* Callback to handle the disconnecion
                                             of the remote peer*/

    struct {
        /* Callback to fill the user's private data */
        uct_sockaddr_priv_pack_callback_t  priv_pack_cb;
        union {
            struct {
                /* On the client side - callback to process an incoming
                 * connection response from the server */
                uct_ep_client_connect_cb_t connect_cb;
            } client;
            struct {
                /* On the server side - callback to process an incoming connection
                 * establishment acknowledgment from the client */
                uct_ep_server_connect_cb_t connect_cb;
            } server;
        };
    } wireup;
} uct_rdmacm_cm_ep_t;


UCS_CLASS_DECLARE(uct_rdmacm_ep_t, const uct_ep_params_t *);
UCS_CLASS_DECLARE_NEW_FUNC(uct_rdmacm_cm_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_rdmacm_cm_ep_t, uct_ep_t);

ucs_status_t uct_rdmacm_cep_disconnect(uct_ep_h ep, unsigned flags);
