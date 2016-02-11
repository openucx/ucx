/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucp_request.h"

#include <ucp/core/ucp_worker.h>
#include <ucp/dt/dt_generic.h>


static UCS_F_ALWAYS_INLINE size_t
ucp_request_generic_dt_pack(ucp_request_t *req, void *dest, size_t length)
{
    ucp_dt_generic_t *dt = ucp_dt_generic(req->send.datatype);
    return dt->ops.pack(req->send.state.dt.generic.state,
                        req->send.state.offset, dest, length);
}

static UCS_F_ALWAYS_INLINE void
ucp_request_generic_dt_finish(ucp_request_t *req)
{
    ucp_dt_generic_t *dt = ucp_dt_generic(req->send.datatype);
    return dt->ops.finish(req->send.state.dt.generic.state);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_request_send_buffer_reg(ucp_request_t *req)
{
    ucs_status_t status;

    status = uct_pd_mem_reg(ucp_ep_pd(req->send.ep), (void*)req->send.buffer,
                            req->send.length, &req->send.state.dt.contig.memh);
    if (status != UCS_OK) {
        ucs_error("failed to register user buffer: %s",
                  ucs_status_string(status));
    }
    return status;
}

static UCS_F_ALWAYS_INLINE void
ucp_request_send_buffer_dereg(ucp_request_t *req)
{
    (void)uct_pd_mem_dereg(ucp_ep_pd(req->send.ep), req->send.state.dt.contig.memh);
}
