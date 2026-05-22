/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_RMA_RNDV_H_
#define UCP_RMA_RNDV_H_

#include <ucp/core/ucp_types.h>
#include <ucp/rndv/rndv.h>


typedef struct {
    ucp_rndv_rts_hdr_t super;
    uint64_t           address;
    ucs_sys_device_t   sys_dev;
    ucs_memory_type_t  mem_type;
} UCS_S_PACKED ucp_rma_rndv_rts_hdr_t;


ucs_status_t ucp_rma_rndv_process_rts(ucp_worker_h worker,
                                      const ucp_rma_rndv_rts_hdr_t *rts,
                                      size_t length);

#endif
