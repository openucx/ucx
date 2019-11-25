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
    struct rdma_cm_id      *id; /* The rdmacm id that is created per this ep */
    struct ibv_cq          *cq; /* Dummy cq used for creating a dummy qp */
    struct ibv_qp          *qp; /* Dummy qp used for generating a unique qp_num */
    void                   *user_data;    /* User data associated with the endpoint */
    uct_ep_disconnect_cb_t disconnect_cb; /* Callback to handle the disconnection
                                             of the remote peer */
    ucs_queue_head_t       ops;
    uint8_t                flags;
    ucs_status_t           status;

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

enum {
    UCT_RDMACM_CM_EP_ON_CLIENT            = UCS_BIT(0),
    UCT_RDMACM_CM_EP_ON_SERVER            = UCS_BIT(1),
    UCT_RDMACM_CM_EP_CONNECTED            = UCS_BIT(2),
    UCT_RDMACM_CM_EP_DISCONNECTING        = UCS_BIT(3), /* uct_ep_disconnect was
                                                           called on the ep */
    UCT_RDMACM_CM_EP_GOT_DISCONNECT_EVENT = UCS_BIT(4),
    UCT_RDMACM_CM_EP_FAILED               = UCS_BIT(5)  /* the ep got err during
                                                           connection
                                                           establishment, e.g.
                                                           served rejected */
};

UCS_CLASS_DECLARE_NEW_FUNC(uct_rdmacm_cm_ep_t, uct_ep_t,
                           const uct_ep_params_t *);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_rdmacm_cm_ep_t, uct_ep_t);

static UCS_F_ALWAYS_INLINE
uct_rdmacm_cm_t *uct_rdmacm_cm_ep_get_cm(uct_rdmacm_cm_ep_t *cep)
{
    /* return the rdmacm connection manager this ep is using */
    return ucs_container_of(cep->super.super.iface, uct_rdmacm_cm_t,
                            super.iface);
}

static UCS_F_ALWAYS_INLINE ucs_async_context_t *
uct_rdmacm_cm_ep_get_async(uct_rdmacm_cm_ep_t *cep)
{
    uct_rdmacm_cm_t *cm      = uct_rdmacm_cm_ep_get_cm(cep);
    uct_priv_worker_t *wpriv = ucs_derived_of(cm->super.iface.worker,
                                              uct_priv_worker_t);

    return wpriv->async;
}

ucs_status_t uct_rdmacm_cm_ep_disconnect(uct_ep_h ep, unsigned flags);

ucs_status_t uct_rdmacm_cm_ep_flush(uct_ep_h ep, unsigned flags,
                                    uct_completion_t *comp);

ucs_status_t
uct_rdamcm_cm_ep_set_qp_num(struct rdma_conn_param *conn_param,
                            uct_rdmacm_cm_ep_t *cep);

ucs_status_t
uct_rdmacm_cm_ep_conn_param_init(uct_rdmacm_cm_ep_t *cep,
                                 struct rdma_conn_param *conn_param);

void uct_rdmacm_cm_ep_client_connect_cb(uct_rdmacm_cm_ep_t *cep,
                                        uct_cm_remote_data_t *remote_data,
                                        ucs_status_t error);

void uct_rdmacm_cm_ep_server_connect_cb(uct_rdmacm_cm_ep_t *cep,
                                        ucs_status_t status);

void uct_rdmacm_cm_ep_error_cb(uct_rdmacm_cm_ep_t *cep,
                               uct_cm_remote_data_t *remote_data,
                               ucs_status_t status);

const char* uct_rdmacm_cm_ep_str(uct_rdmacm_cm_ep_t *cep, char *str,
                                 size_t max_len);

void uct_rdmacm_cm_ep_invoke_completions(uct_rdmacm_cm_ep_t *ep,
                                         ucs_status_t status);
