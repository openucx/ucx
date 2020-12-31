/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef WIREUP_CM_H_
#define WIREUP_CM_H_

#include <uct/api/uct.h>
#include <ucp/api/ucp.h>
#include <ucp/core/ucp_types.h>


typedef struct ucp_cm_client_connect_progress_arg {
    ucp_ep_h                   ucp_ep;
    ucp_wireup_sockaddr_data_t *sa_data;
    uct_device_addr_t          *dev_addr;
} ucp_cm_client_connect_progress_arg_t;


unsigned ucp_cm_ep_init_flags(const ucp_worker_h worker,
                              const ucp_ep_params_t *params);

int ucp_ep_init_flags_has_cm(unsigned ep_init_flags);

ucs_status_t ucp_ep_cm_connect_server_lane(ucp_ep_h ep,
                                           uct_listener_h uct_listener,
                                           uct_conn_request_h uct_conn_req,
                                           ucp_rsc_index_t cm_idx);

ucs_status_t ucp_ep_client_cm_connect_start(ucp_ep_h ucp_ep,
                                            const ucp_ep_params_t *params);

ucs_status_t ucp_ep_client_cm_create_uct_ep(ucp_ep_h ucp_ep);

int ucp_cm_server_conn_request_progress_cb_pred(const ucs_callbackq_elem_t *elem,
                                                void *arg);

void ucp_cm_server_conn_request_cb(uct_listener_h listener, void *arg,
                                   const uct_cm_listener_conn_request_args_t
                                   *conn_req_args);

ucs_status_t
ucp_ep_cm_server_create_connected(ucp_worker_h worker, unsigned ep_init_flags,
                                  const ucp_unpacked_address_t *remote_addr,
                                  ucp_conn_request_h conn_request,
                                  ucp_ep_h *ep_p);

void ucp_ep_cm_disconnect_cm_lane(ucp_ep_h ucp_ep);

ucp_request_t* ucp_ep_cm_close_request_get(ucp_ep_h ep,
                                           const ucp_request_param_t *param);

void ucp_ep_cm_slow_cbq_cleanup(ucp_ep_h ep);

#endif /* WIREUP_CM_H_ */
