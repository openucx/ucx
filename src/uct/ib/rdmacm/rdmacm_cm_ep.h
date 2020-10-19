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
    uct_cm_base_ep_t  super;
    struct rdma_cm_id *id;  /* The rdmacm id that is created per this ep */
    struct ibv_cq     *cq;  /* Dummy cq used for creating a dummy qp */
    struct ibv_qp     *qp;  /* Dummy qp used for generating a unique qp_num */
    uint8_t           flags;
    ucs_status_t      status;
} uct_rdmacm_cm_ep_t;

enum {
    UCT_RDMACM_CM_EP_ON_CLIENT                = UCS_BIT(0),
    UCT_RDMACM_CM_EP_ON_SERVER                = UCS_BIT(1),
    UCT_RDMACM_CM_EP_CLIENT_CONN_CB_INVOKED   = UCS_BIT(2), /* Connect callback was
                                                               invoked on the client. */
    UCT_RDMACM_CM_EP_SERVER_NOTIFY_CB_INVOKED = UCS_BIT(3), /* Notify callback was
                                                               invoked on the server. */
    UCT_RDMACM_CM_EP_GOT_DISCONNECT           = UCS_BIT(4), /* Got disconnect event. */
    UCT_RDMACM_CM_EP_DISCONNECTING            = UCS_BIT(5), /* @ref uct_ep_disconnect was
                                                               called on the ep. */
    UCT_RDMACM_CM_EP_FAILED                   = UCS_BIT(6)  /* The EP is in error state,
                                                               see @ref
                                                               uct_rdmacm_cm_ep_t::status.*/
};


static UCS_F_ALWAYS_INLINE
uct_rdmacm_cm_t *uct_rdmacm_cm_ep_get_cm(uct_rdmacm_cm_ep_t *cep)
{
    /* return the rdmacm connection manager this ep is using */
    return ucs_container_of(cep->super.super.super.iface, uct_rdmacm_cm_t,
                            super.iface);
}


static UCS_F_ALWAYS_INLINE
ucs_async_context_t *uct_rdmacm_cm_ep_get_async(uct_rdmacm_cm_ep_t *cep)
{
    return uct_rdmacm_cm_get_async(uct_rdmacm_cm_ep_get_cm(cep));
}

UCS_CLASS_DECLARE_NEW_FUNC(uct_rdmacm_cm_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_rdmacm_cm_ep_t, uct_ep_t);

ucs_status_t uct_rdmacm_cm_ep_disconnect(uct_ep_h ep, unsigned flags);

ucs_status_t uct_rdmacm_cm_ep_conn_notify(uct_ep_h ep);

ucs_status_t uct_rdmacm_cm_ep_pack_cb(uct_rdmacm_cm_ep_t *cep,
                                      struct rdma_conn_param *conn_param);

ucs_status_t
uct_rdamcm_cm_ep_set_qp_num(struct rdma_conn_param *conn_param,
                            uct_rdmacm_cm_ep_t *cep);

void uct_rdmacm_cm_ep_error_cb(uct_rdmacm_cm_ep_t *cep,
                               uct_cm_remote_data_t *remote_data,
                               ucs_status_t status);

void uct_rdmacm_cm_ep_set_failed(uct_rdmacm_cm_ep_t *cep,
                                 uct_cm_remote_data_t *remote_data,
                                 ucs_status_t status);

const char* uct_rdmacm_cm_ep_str(uct_rdmacm_cm_ep_t *cep, char *str,
                                 size_t max_len);

int uct_rdmacm_ep_is_connected(uct_rdmacm_cm_ep_t *cep);

void uct_rdmacm_cm_ep_client_connect_cb(uct_rdmacm_cm_ep_t *cep,
                                        uct_cm_remote_data_t *remote_data,
                                        ucs_status_t status);

void uct_rdmacm_cm_ep_server_conn_notify_cb(uct_rdmacm_cm_ep_t *cep,
                                            ucs_status_t status);
