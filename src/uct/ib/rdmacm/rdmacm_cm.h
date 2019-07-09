/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#include <uct/base/uct_cm.h>
#include <ucs/sys/sock.h>
#include "rdmacm_def.h"

#define UCT_RDMACM_CM_MAX_CONN_PRIV \
        (UCT_RDMACM_TCP_PRIV_DATA_LEN) - (sizeof(uct_rdmacm_priv_data_hdr_t))

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

ucs_status_t uct_rdmacm_cm_open(uct_component_h component, uct_worker_h worker,
                                uct_cm_h *uct_cm_p);
