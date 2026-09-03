/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_RMA_RNDV_H_
#define UCP_RMA_RNDV_H_

#include <ucp/core/ucp_types.h>
#include <ucp/rndv/rndv.h>


/* First release whose RMA/RNDV completion header includes operation flags. */
#define UCP_PROTO_RMA_RNDV_MIN_DST_VERSION 24

typedef struct {
    ucp_rndv_rts_hdr_t super;
    uint64_t           address;
    ucs_sys_device_t   sys_dev;
    ucs_memory_type_t  mem_type;
} UCS_S_PACKED ucp_rma_rndv_rts_hdr_t;


static UCS_F_ALWAYS_INLINE int
ucp_proto_rma_rndv_is_peer_supported(unsigned dst_version)
{
    return dst_version >= UCP_PROTO_RMA_RNDV_MIN_DST_VERSION;
}

static UCS_F_ALWAYS_INLINE int
ucp_proto_rma_rndv_is_err_mode_supported(ucp_err_handling_mode_t err_mode)
{
    return err_mode != UCP_ERR_HANDLING_MODE_FAILOVER;
}



ucs_status_t ucp_rma_rndv_process_rts(ucp_worker_h worker,
                                      const ucp_rma_rndv_rts_hdr_t *rts,
                                      size_t length);

ucp_request_t *ucp_rma_rndv_flush_open(ucp_request_t *rndv_req);

void ucp_rma_rndv_flush_close(ucp_request_t *recv_req, ucp_ep_h ep,
                              ucs_status_t status);

#endif
