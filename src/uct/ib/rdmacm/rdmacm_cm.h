/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#include <uct/base/uct_cm.h>

typedef struct uct_rdmacm_cm {
    uct_cm_t                  super;
    struct rdma_event_channel *ev_ch;
    uct_worker_h              worker;
} uct_rdmacm_cm_t;


typedef struct uct_rdmacm_listener {
    uct_listener_t                       super;
    struct rdma_cm_id                    *id;
    uct_listener_conn_request_callback_t conn_request_cb;
    void                                 *user_data;
} uct_rdmacm_listener_t;


typedef struct uct_rdmacm_cep {
    uct_base_ep_t          super;
    uct_rdmacm_cm_t        *cm;
    struct rdma_cm_id      *id;
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

ucs_status_t uct_rdmacm_cm_open(uct_component_h component, uct_worker_h worker,
                                uct_cm_h *uct_cm_p);
