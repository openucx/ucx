/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucp/rma/rma_bw.inl>

int ucp_rma_bw_test_detach_pending(void)
{
    ucp_request_t req          = {0};
    ucp_rma_bw_sample_t sample = {0};
    ucp_worker_t worker        = {0};
    ucp_rma_bw_frag_t *frag;
    uct_completion_t *comp;

    req.flags               = UCP_REQUEST_FLAG_RMA_BW_SAMPLE;
    req.send.rma.bw_sample  = &sample;
    sample.req              = &req;
    sample.worker           = &worker;
    sample.in_use           = 1;
    worker.rma_bw_active    = 1;

    comp = ucp_rma_bw_frag_start(&req, 0, &frag);
    ucp_rma_bw_frag_posted(frag, 4096, UCS_INPROGRESS);
    req.send.state.uct_comp.count = 2;
    ucp_rma_bw_abort(&req, UCS_ERR_CANCELED);
    ucp_rma_bw_sample_detach(&req);

    if ((req.flags & UCP_REQUEST_FLAG_RMA_BW_SAMPLE) ||
        (req.send.rma.bw_sample != NULL) || !sample.in_use ||
        !sample.invalid || (worker.rma_bw_active != 1)) {
        return 1;
    }

    ucp_invoke_uct_completion(comp, UCS_OK);
    return sample.in_use || (worker.rma_bw_active != 0);
}
