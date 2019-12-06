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
    uct_ep_h                   uct_cm_ep;
    ucp_wireup_sockaddr_data_t *sa_data;
    uct_device_addr_t          *dev_addr;
} ucp_cm_client_connect_progress_arg_t;


typedef struct ucp_cm_disconnect_progress_arg {
    ucp_ep_h                   ucp_ep;
    uct_ep_h                   uct_cm_ep;
} ucp_cm_disconnect_progress_arg_t;


unsigned ucp_cm_ep_init_flags(const ucp_worker_h worker,
                              const ucp_ep_params_t *params);

ucs_status_t ucp_ep_cm_connect_server_lane(ucp_ep_h ep,
                                           ucp_conn_request_h conn_request);

unsigned
ucp_cm_ep_init_flags(const ucp_worker_h worker, const ucp_ep_params_t *params);

ucs_status_t ucp_ep_client_cm_connect_start(ucp_ep_h ucp_ep,
                                            const ucp_ep_params_t *params);

void ucp_cm_server_conn_request_cb(uct_listener_h listener, void *arg,
                                   const char *local_dev_name,
                                   uct_conn_request_h conn_request,
                                   const uct_cm_remote_data_t *remote_data);

ucs_status_t
ucp_ep_cm_server_create_connected(ucp_worker_h worker, unsigned ep_init_flags,
                                  const ucp_unpacked_address_t *remote_addr,
                                  ucp_conn_request_h conn_request,
                                  ucp_ep_h *ep_p);

void ucp_ep_cm_disconnect_cm_lane(ucp_ep_h ucp_ep);

ucp_request_t* ucp_ep_cm_close_request_get(ucp_ep_h ep);

#endif /* WIREUP_CM_H_ */
