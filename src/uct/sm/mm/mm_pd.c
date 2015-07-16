/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "mm_pd.h"


ucs_status_t uct_mm_mem_alloc(uct_pd_h pd, size_t *length_p, void **address_p,
                              uct_mem_h *memh_p UCS_MEMTRACK_ARG)
{
    ucs_status_t status;
    uct_mm_seg_t *seg;

    seg = ucs_malloc(sizeof(*seg), "mm_seg");
    if (NULL == seg) {
        ucs_error("Failed to allocate memory for mm segment");
        return UCS_ERR_NO_MEMORY;
    }

    status = uct_mm_pd_mapper_ops(pd)->alloc(length_p, UCS_TRY, &seg->address,
                                             &seg->mmid UCS_MEMTRACK_VAL);
    if (status != UCS_OK) {
        ucs_free(seg);
        return status;
    }

    *address_p = seg->address;
    *memh_p    = seg;

    ucs_debug("mm allocated address %p length %zu mmid %"PRIu64,
              *address_p, *length_p, seg->mmid);
    return UCS_OK;
}

ucs_status_t uct_mm_mem_free(uct_pd_h pd, uct_mem_h memh)
{
    uct_mm_seg_t *seg = memh;
    ucs_status_t status;

    status = uct_mm_pd_mapper_ops(pd)->release(seg->address);
    if (status != UCS_OK) {
        return status;
    }

    ucs_free(seg);
    return UCS_OK;
}

ucs_status_t uct_mm_pd_query(uct_pd_h pd, uct_pd_attr_t *pd_attr)
{
    pd_attr->cap.flags     = UCT_PD_FLAG_ALLOC;
    pd_attr->cap.max_alloc = ULONG_MAX;
    pd_attr->cap.max_reg   = 0;
    memset(&pd_attr->local_cpus, 0xff, sizeof(pd_attr->local_cpus));
    return UCS_OK;
}

ucs_status_t uct_mm_mkey_pack(uct_pd_h pd, uct_mem_h memh, void *rkey_buffer)
{
    uct_mm_packed_rkey_t *rkey = rkey_buffer;
    uct_mm_seg_t *seg = memh;

    rkey->mmid      = seg->mmid;
    rkey->owner_ptr = (uintptr_t)seg->address;
    ucs_trace("packed rkey: mmid %"PRIu64" owner_ptr %"PRIxPTR,
              rkey->mmid, rkey->owner_ptr);
    return UCS_OK;
}

ucs_status_t uct_mm_rkey_unpack(uct_pd_component_t *pdc, const void *rkey_buffer,
                                uct_rkey_t *rkey_p, void **handle_p)
{
    /* user is responsible to free rkey_buffer */
    const uct_mm_packed_rkey_t *rkey = rkey_buffer;
    ucs_status_t status;
    void *client_ptr;

    ucs_trace("unpacking rkey: mmid %"PRIu64" owner_ptr %"PRIxPTR,
              rkey->mmid, rkey->owner_ptr);

    status = uct_mm_pdc_mapper_ops(pdc)->attach(rkey->mmid, &client_ptr);
    if (status != UCS_OK) {
        return status;
    }

    /* store the offset of the addresses, this can be used directly to translate
     * the remote VA to local VA of the attached segment */
    *handle_p = client_ptr;
    *rkey_p   = (uintptr_t)client_ptr - rkey->owner_ptr;
    return UCS_OK;
}

ucs_status_t uct_mm_rkey_release(uct_pd_component_t *pdc, uct_rkey_t rkey, void *handle)
{
    void *client_ptr = handle;
    return uct_mm_pdc_mapper_ops(pdc)->release(client_ptr);
}
