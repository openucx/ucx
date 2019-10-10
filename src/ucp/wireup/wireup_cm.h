/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef WIREUP_CM_H_
#define WIREUP_CM_H_

#include <uct/api/uct.h>
#include <ucp/api/ucp.h>


ucs_status_t ucp_ep_client_cm_connect_start(ucp_ep_h ucp_ep,
                                            const ucp_ep_params_t *params);

void ucp_cm_server_conn_request_cb(uct_listener_h listener, void *arg,
                                   const char *local_dev_name,
                                   uct_conn_request_h conn_request,
                                   const uct_cm_remote_data_t *remote_data);

#endif /* WIREUP_CM_H_ */
