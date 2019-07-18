/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "rdmacm_listener.h"

typedef struct uct_rdmacm_cep {
    uct_base_ep_t          super;
    uct_rdmacm_cm_t        *cm;
    struct rdma_cm_id      *id;
    struct ibv_cq          *cq; /* dummy cq used for creating a dummy qp */
    struct ibv_qp          *qp; /* dummy qp used for generating a unique qp_num */
    void                   *user_data;
    uct_ep_disconnect_cb_t disconnect_cb;

    struct {
        uct_sockaddr_priv_pack_callback_t priv_pack_cb;
        union {
            struct {
                uct_ep_client_connect_cb_t connect_cb;
            } client;
            struct {
                uct_ep_server_connect_cb_t connect_cb;
            } server;
        };
    } wireup;
} uct_rdmacm_cep_t;


UCS_CLASS_DECLARE(uct_rdmacm_ep_t, const uct_ep_params_t *);
UCS_CLASS_DECLARE_NEW_FUNC(uct_rdmacm_cep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_rdmacm_cep_t, uct_ep_t);

ucs_status_t uct_rdmacm_cep_disconnect(uct_ep_h ep, unsigned flags);
