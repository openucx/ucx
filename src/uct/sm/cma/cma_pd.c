/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "cma_pd.h"

ucs_status_t uct_cma_pd_query(uct_pd_h pd, uct_pd_attr_t *pd_attr)
{
    pd_attr->rkey_packed_size  = sizeof(uct_cma_packed_rkey_t);
    pd_attr->cap.flags         = UCT_PD_FLAG_REG;
    pd_attr->cap.max_alloc     = 0;
    pd_attr->cap.max_reg       = ULONG_MAX;

    memset(&pd_attr->local_cpus, 0xff, sizeof(pd_attr->local_cpus));
    return UCS_OK;
}

ucs_status_t uct_cma_mkey_pack(uct_pd_h pd, uct_mem_h memh, void *rkey_buffer)
{
    uct_cma_packed_rkey_t *rkey = rkey_buffer;

    rkey->cma_id      = getpid();
    ucs_trace("packed rkey: pid %d", rkey->cma_id);
    return UCS_OK;
}

ucs_status_t uct_cma_rkey_unpack(uct_pd_component_t *pdc, const void *rkey_buffer,
                                uct_rkey_t *rkey_p, void **handle_p)
{
    /* user is responsible to free rkey_buffer */
    const uct_cma_packed_rkey_t *rkey = rkey_buffer;

    ucs_trace("unpacking rkey: pid %d", rkey->cma_id);

    *handle_p = NULL;
    *rkey_p   = (uintptr_t)rkey->cma_id;
    return UCS_OK;
}

ucs_status_t uct_cma_rkey_release(uct_pd_component_t *pdc, uct_rkey_t rkey, void *handle)
{
    /* nop */
    return UCS_OK;
}

ucs_status_t uct_cma_mem_reg(uct_pd_h pd, void *address, size_t length,
                                     uct_mem_h *memh_p)
{
    return UCS_OK;
}

ucs_status_t uct_cma_mem_dereg(uct_pd_h pd, uct_mem_h memh)
{
    return UCS_OK;
}
