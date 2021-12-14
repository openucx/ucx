/*
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#include <ucp/api/ucp.h>

extern void ucxgo_completeGoSendRequest(void *request, ucs_status_t status, void *callback_id);

extern void ucxgo_completeGoTagRecvRequest(void *request, ucs_status_t status, ucp_tag_recv_info_t *info, void *callback_id);

extern void ucxgo_completeGoErrorHandler(void* arg, ucp_ep_h ep, ucs_status_t status);

extern void ucxgo_completeConnHandler(ucp_conn_request_h conn_request, void *callback_id);

extern ucs_status_t ucxgo_amRecvCallback(void *callback_id, void *header, size_t header_length,
                                         void *data, size_t length, ucp_am_recv_param_t *param);

extern void ucxgo_completeAmRecvData(void *request, ucs_status_t status, size_t length, void *callback_id);