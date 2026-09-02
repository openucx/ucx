/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef TEST_UCP_H
#define TEST_UCP_H

#include "ucp/api/ucp.h"

typedef struct {
    ucp_context_h context;
    ucp_worker_h  worker;
    ucp_ep_h      ep;
} ucp_t;


typedef struct {
    ucp_rkey_h rkey;
    uint64_t   remote_address;
} rkey_t;


ucp_t create_ucp();


void destroy_ucp(ucp_t);


ucp_mem_h send_rkey(int, void*, size_t, ucp_context_h);


rkey_t recv_rkey(int, ucp_ep_h);

#endif
