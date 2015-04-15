/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "ucp_int.h"


ucs_status_t ucp_mem_map(ucp_context_h context, void **address_p, size_t length,
                         unsigned flags, ucp_mem_h *memh_p)
{
    return UCS_ERR_NOT_IMPLEMENTED;
}

ucs_status_t ucp_mem_unmap(ucp_context_h context, ucp_mem_h memh)
{
    return UCS_ERR_NOT_IMPLEMENTED;
}

ucs_status_t ucp_rkey_pack(ucp_mem_h memh, void **rkey_buffer_p, size_t *size_p)
{
    return UCS_ERR_NOT_IMPLEMENTED;
}

ucs_status_t ucp_rkey_unpack(ucp_context_h context, void *rkey_buffer,
                             ucp_rkey_h *rkey_p)
{
    return UCS_ERR_NOT_IMPLEMENTED;
}

ucs_status_t ucp_rkey_destroy(ucp_rkey_h rk_rkey_destrey)
{
    return UCS_ERR_NOT_IMPLEMENTED;
}

ucs_status_t ucp_rma_put(ucp_ep_h ep, const void *buffer, size_t length,
                         uint64_t remote_addr, ucp_rkey_h rkey)
{
    return UCS_ERR_UNSUPPORTED;
}

ucs_status_t ucp_rma_get(ucp_ep_h ep, void *buffer, size_t length,
                         uint64_t remote_addr, ucp_rkey_h rkey)
{
    return UCS_ERR_UNSUPPORTED;
}

ucs_status_t ucp_rma_fence(ucp_context_h context)
{
    return UCS_ERR_UNSUPPORTED;
}

ucs_status_t ucp_rma_flush(ucp_context_h context)
{
    return UCS_ERR_UNSUPPORTED;
}
