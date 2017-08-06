/**
 * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_worker.h>
#include <ucp/core/ucp_context.h>

#include <ucs/debug/profile.h>


UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_stream_send_nb,
                 (ep, buffer, count, datatype, cb, flags),
                 ucp_ep_h ep, const void *buffer, size_t count,
                 uintptr_t datatype, ucp_send_callback_t cb, unsigned flags)
{
    return UCS_STATUS_PTR(UCS_OK);
}
