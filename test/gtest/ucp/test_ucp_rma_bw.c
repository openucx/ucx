/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucp/rma/rma_bw.inl>

#define RMA_BW_CHECK(_cond) do { if (!(_cond)) return __LINE__; } while (0)

static void ucp_rma_bw_test_complete(uct_completion_t *comp)
{
    ucp_request_t *req = ucs_container_of(comp, ucp_request_t,
                                          send.state.uct_comp);
    ++*(int*)req->user_data;
}

static void ucp_rma_bw_test_init(ucp_request_t *req,
                                 ucp_rma_bw_sample_t *sample, int *completed)
{
    memset(req, 0, sizeof(*req));
    memset(sample, 0, sizeof(*sample));
    *completed                      = 0;
    req->user_data                  = completed;
    req->flags                      = UCP_REQUEST_FLAG_RMA_BW_SAMPLE;
    req->send.rma.bw_sample         = sample;
    req->send.state.uct_comp.func   = ucp_rma_bw_test_complete;
    req->send.state.uct_comp.count  = 1;
    req->send.state.uct_comp.status = UCS_OK;
    sample->req                     = req;
}

int ucp_rma_bw_test_async_sync_retry(void)
{
    ucp_request_t req;
    ucp_rma_bw_sample_t sample;
    ucp_rma_bw_frag_t *frag;
    uct_completion_t *comp;
    int completed;

    ucp_rma_bw_test_init(&req, &sample, &completed);
    comp = ucp_rma_bw_frag_start(&req, 0, &frag);
    RMA_BW_CHECK(frag != NULL);
    ucp_rma_bw_frag_posted(frag, 1024, UCS_ERR_NO_RESOURCE);
    RMA_BW_CHECK(sample.lanes[0].bytes == 0);
    RMA_BW_CHECK(sample.num_frags == 0);

    comp = ucp_rma_bw_frag_start(&req, 0, &frag);
    ucp_rma_bw_frag_posted(frag, 4096, UCS_INPROGRESS);
    ++req.send.state.uct_comp.count;
    RMA_BW_CHECK(sample.lanes[0].bytes == 4096);
    RMA_BW_CHECK(sample.num_frags == 1);

    ucp_rma_bw_frag_start(&req, 1, &frag);
    ucp_rma_bw_frag_posted(frag, 8192, UCS_OK);
    RMA_BW_CHECK(sample.lanes[1].bytes == 8192);
    RMA_BW_CHECK(sample.lanes[1].num_async == 0);

    ucp_invoke_uct_completion(comp, UCS_OK);
    RMA_BW_CHECK(req.send.state.uct_comp.count == 1);
    RMA_BW_CHECK(completed == 0);
    ucp_invoke_uct_completion(&req.send.state.uct_comp, UCS_OK);
    RMA_BW_CHECK(completed == 1);
    RMA_BW_CHECK(req.send.state.uct_comp.status == UCS_OK);
    RMA_BW_CHECK(sample.lanes[0].last_comp > sample.lanes[0].first_post);
    return 0;
}

int ucp_rma_bw_test_abort(void)
{
    ucp_request_t req;
    ucp_rma_bw_sample_t sample;
    ucp_rma_bw_frag_t *frag;
    uct_completion_t *comp;
    int completed;

    ucp_rma_bw_test_init(&req, &sample, &completed);
    comp = ucp_rma_bw_frag_start(&req, 0, &frag);
    ucp_rma_bw_frag_posted(frag, 4096, UCS_INPROGRESS);
    ++req.send.state.uct_comp.count;

    ucp_rma_bw_abort(&req, UCS_ERR_CANCELED);
    RMA_BW_CHECK(sample.invalid);
    RMA_BW_CHECK(completed == 0);
    ucp_invoke_uct_completion(comp, UCS_OK);
    RMA_BW_CHECK(completed == 1);
    RMA_BW_CHECK(req.send.state.uct_comp.status == UCS_ERR_CANCELED);
    return 0;
}

int ucp_rma_bw_test_frag_cap(void)
{
    ucp_request_t req;
    ucp_rma_bw_sample_t sample;
    ucp_rma_bw_frag_t *frag;
    uct_completion_t *comp;
    int completed;

    ucp_rma_bw_test_init(&req, &sample, &completed);
    sample.num_frags = UCP_RMA_BW_MAX_FRAGS;
    comp = ucp_rma_bw_frag_start(&req, 0, &frag);
    RMA_BW_CHECK(comp == &req.send.state.uct_comp);
    RMA_BW_CHECK(frag == NULL);
    RMA_BW_CHECK(sample.invalid);
    return 0;
}


int ucp_rma_bw_test_transport_error(void)
{
    ucp_request_t req;
    ucp_rma_bw_sample_t sample;
    ucp_rma_bw_frag_t *frag;
    uct_completion_t *comp0, *comp1;
    int completed;

    ucp_rma_bw_test_init(&req, &sample, &completed);
    comp0 = ucp_rma_bw_frag_start(&req, 0, &frag);
    ucp_rma_bw_frag_posted(frag, 4096, UCS_INPROGRESS);
    ++req.send.state.uct_comp.count;
    comp1 = ucp_rma_bw_frag_start(&req, 1, &frag);
    ucp_rma_bw_frag_posted(frag, 4096, UCS_INPROGRESS);
    ++req.send.state.uct_comp.count;

    ucp_invoke_uct_completion(&req.send.state.uct_comp, UCS_OK);
    RMA_BW_CHECK(completed == 0);
    ucp_invoke_uct_completion(comp1, UCS_ERR_IO_ERROR);
    RMA_BW_CHECK(sample.invalid);
    RMA_BW_CHECK(completed == 0);
    ucp_invoke_uct_completion(comp0, UCS_OK);
    RMA_BW_CHECK(completed == 1);
    RMA_BW_CHECK(req.send.state.uct_comp.status == UCS_ERR_IO_ERROR);
    return 0;
}
