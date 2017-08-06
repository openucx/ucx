/**
 * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_worker.h>
#include <ucp/core/ucp_context.h>

#include <ucs/debug/profile.h>


UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_stream_recv_data_nb,
                 (ep, length), ucp_ep_h ep, size_t *length)
{
    return UCS_STATUS_PTR(UCS_OK);
}

void ucp_stream_data_release(ucp_ep_h ep, void *data)
{
}

static ucs_status_t ucp_eager_only_handler(void *arg, void *data, size_t length,
                                           unsigned am_flags)
{
    return UCS_ERR_NOT_IMPLEMENTED;
}

static void ucp_eager_dump(ucp_worker_h worker, uct_am_trace_type_t type,
                           uint8_t id, const void *data, size_t length,
                           char *buffer, size_t max)
{
    /* TODO: */
}

UCP_DEFINE_AM(UCP_FEATURE_STREAM, UCP_AM_ID_STREAM_EAGER_ONLY, ucp_eager_only_handler,
              ucp_eager_dump, UCT_CB_FLAG_SYNC);
