/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * Copyright (c) Los Alamos National Security, LLC. 2016.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "mm_pd.h"
#include "mm_iface.h"

#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>
#include "xpmem.h"

static ucs_status_t uct_xpmem_query()
{
    int ver;

    ver = xpmem_version();
    if (ver < 0) {
        ucs_debug("Failed to query XPMEM version %d, %m", ver);
        return UCS_ERR_UNSUPPORTED;
    }

    return UCS_OK;
}

static size_t uct_xpmem_get_path_size(uct_pd_h pd)
{
    return 0;
}

static ucs_status_t uct_xmpem_reg(void *address, size_t size, 
                                  uct_mm_id_t *mmid_p)
{
    xpmem_segid_t segid; /* 64bit ID */
    const size_t page_size = ucs_get_page_size();
    void*  addr_aligned   = (void *)ucs_align_down((uintptr_t)address, page_size);
    off_t  diff = (uintptr_t)address - (uintptr_t)addr_aligned;
    size_t length_aligned = ucs_align_up(diff + size, page_size);

    ucs_assert_always(address >= addr_aligned);

    segid = xpmem_make(addr_aligned, length_aligned, XPMEM_PERMIT_MODE, (void *)0666);
    if (segid < 0) {
        ucs_error("Failed to register memory with xpmem (addr: %p, len: %zu): %m",
                  address, size);
        return UCS_ERR_IO_ERROR;
    }

    *mmid_p = segid;

    ucs_debug("Calling reg for address %p cookie %lu", address, *mmid_p);
    return UCS_OK;
}

static ucs_status_t uct_xpmem_dereg(uct_mm_id_t mmid)
{
    int rc;
    xpmem_segid_t segid = (xpmem_segid_t)mmid;

    ucs_debug("Calling dereg for cookie %lu", mmid);

    ucs_assert_always(segid > 0);
    rc = xpmem_remove(segid);
    if (rc < 0) {
        /* No error since there a chance that it already
         * was released or deregistered */
        ucs_debug("Failed to de-register memory: %m");
    }

    return UCS_OK;
}

static ucs_status_t uct_xpmem_attach(uct_mm_id_t mmid, size_t length, 
                                     void *remote_address, void **local_address,
                                     uint64_t *cookie, const char *path)
{
    xpmem_segid_t segid = (xpmem_segid_t)mmid;
    xpmem_apid_t apid;
    struct xpmem_addr addr;
    ucs_status_t status;
    const size_t page_size = ucs_get_page_size();
    void*  addr_aligned  = (void *)ucs_align_down((uintptr_t)remote_address, page_size);
    off_t  diff = (uintptr_t)remote_address - (uintptr_t)addr_aligned;

    ucs_debug("Calling attach for address %p", remote_address);

    apid = xpmem_get(segid, XPMEM_RDWR, XPMEM_PERMIT_MODE, NULL);
    if (apid < 0) {
        ucs_error("Failed to xpmem_get: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_xget;
    }

    addr.apid = apid;
    addr.offset = diff;

    *local_address = xpmem_attach(addr, length, NULL);
    if (*local_address < 0) {
        ucs_error("Failed to xpmem_attach: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_xattach;
    }
    *cookie = apid;

    return UCS_OK;

err_xattach:
    xpmem_release(apid);
err_xget:
    return status;
}

static ucs_status_t uct_xpmem_detach(uct_mm_remote_seg_t *mm_desc)
{
    int rc;

    ucs_debug("Calling detach with address %p cookie %lu",
              mm_desc->address, mm_desc->cookie);

    rc = xpmem_detach(mm_desc->address);
    if (rc < 0) {
        ucs_error("Failed to xpmem_detach: %m");
        return UCS_ERR_IO_ERROR;
    }

    rc = xpmem_release((xpmem_apid_t) mm_desc->cookie);
    if (rc < 0) {
        ucs_error("Failed to xpmem_release: %m");
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

static ucs_status_t uct_xpmem_alloc(uct_pd_h pd, size_t *length_p, ucs_ternary_value_t
                                    hugetlb, void **address_p,
                                    uct_mm_id_t *mmid_p, const char **path_p
                                    UCS_MEMTRACK_ARG)
{
    ucs_status_t status = UCS_ERR_NO_MEMORY;
    const size_t page_size = ucs_get_page_size();
    const size_t length_aligned = ucs_align_up(*length_p, page_size);

    if (0 == *length_p) {
        ucs_error("Unexpected length %zu", *length_p);
        status = UCS_ERR_INVALID_PARAM;
        goto err;
    }

    /* TBD: any ideas for better allocation */
    *address_p = ucs_memalign(page_size, length_aligned, "XPMEM memory");
    if (NULL == *address_p) {
        ucs_error("Failed to allocate %zu bytes of memory", *length_p);
        goto err;
    }

    ucs_debug("Calling alloc with address %p", *address_p);
    status = uct_xmpem_reg(*address_p, length_aligned, mmid_p);
    if (UCS_OK != status) {
        ucs_free(*address_p);
    }
err:
    return status;
}

static ucs_status_t uct_xpmem_free(void *address, uct_mm_id_t mm_id, size_t length,
                                   const char *path)
{
    ucs_status_t status = uct_xpmem_dereg(mm_id);

    if (UCS_OK != status) {
        return status;
    }

    ucs_free(address);
    ucs_debug("Calling free with address %p, cookie %lu", address, mm_id);
    return UCS_OK;
}

static uct_mm_mapper_ops_t uct_xpmem_mapper_ops = {
    .query   = uct_xpmem_query,
    .get_path_size = uct_xpmem_get_path_size,
    .reg     = uct_xmpem_reg,
    .dereg   = uct_xpmem_dereg,
    .alloc   = uct_xpmem_alloc,
    .attach  = uct_xpmem_attach,
    .detach  = uct_xpmem_detach,
    .free    = uct_xpmem_free
};

UCT_MM_COMPONENT_DEFINE(uct_xpmem_pd, "xpmem", &uct_xpmem_mapper_ops, uct, "XPMEM_")
UCT_PD_REGISTER_TL(&uct_xpmem_pd, &uct_mm_tl);
