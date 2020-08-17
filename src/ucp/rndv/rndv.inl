/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef RNDV_INL_
#define RNDV_INL_

static UCS_F_ALWAYS_INLINE int
ucp_rndv_is_get_zcopy(ucp_request_t *req, ucp_context_h context)
{
    return ((context->config.ext.rndv_mode == UCP_RNDV_MODE_GET_ZCOPY) ||
            ((context->config.ext.rndv_mode == UCP_RNDV_MODE_AUTO) &&
             (!UCP_MEM_IS_CUDA(req->send.mem_type) ||
              (req->send.length < context->config.ext.rndv_pipeline_send_thresh))));
}

#endif