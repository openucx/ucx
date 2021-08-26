/*
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#include <ucp/api/ucp.h>

extern void  ucxgo_completeGoSendRequest(void *request, ucs_status_t status, void *calback_id);

extern void  ucxgo_completeGoTagRecvRequest(void *request, ucs_status_t status, ucp_tag_recv_info_t *info, void *calback_id);

extern void  ucxgo_completeGoErrorHandler(ucp_ep_h ep, ucs_status_t status);
