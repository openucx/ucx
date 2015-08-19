/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#include "mm_pd.h"
#include "mm_iface.h"

#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>
#include "xpmem.h"


static ucs_status_t uct_xpmem_query()
{
    int fd, ver;

    fd = open(XPMEM_DEV_PATH, O_RDWR);
    if (fd < 0) {
        ucs_debug("Could not open the XPMEM device file at /dev/xpmem: %m. Disabling xpmem resource");
        return UCS_ERR_UNSUPPORTED;
    }
    close(fd);

    ver = xpmem_version();
    if (ver < 0) {
        ucs_debug("Failed to query XPMEM version %d, %m", ver);
        return UCS_ERR_UNSUPPORTED;
    }

    return UCS_OK;
}

static ucs_status_t uct_xmpem_reg(void *address, size_t size, 
                                  uct_mm_id_t *mmid_p)
{
    xpmem_segid_t segid; /* 64bit ID*/
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
    return UCS_OK;
}

static ucs_status_t uct_xpmem_dereg(uct_mm_id_t mmid)
{
    int rc;
    xpmem_segid_t segid = (xpmem_segid_t)mmid;

    ucs_assert_always(segid > 0);
    rc = xpmem_remove(segid);
    if (rc < 0) {
        ucs_error("Failed to de-register memory: %m");
        return UCS_ERR_IO_ERROR;
    }
    return UCS_OK;
}

static ucs_status_t uct_xpmem_attach(uct_mm_id_t mmid, size_t length, 
                                     void *rem_address, 
                                     uct_mm_mapped_desc_t **mm_desc)
{
    xpmem_segid_t segid = (xpmem_segid_t)mmid;
    xpmem_apid_t apid;
    struct xpmem_addr addr;
    ucs_status_t status;
    const size_t page_size = ucs_get_page_size();
    void*  addr_aligned   = (void *)ucs_align_down((uintptr_t)rem_address, page_size);
    off_t  diff = (uintptr_t)rem_address - (uintptr_t)addr_aligned;

    *mm_desc = ucs_malloc(sizeof(uct_mm_mapped_desc_t), "mm_desc");
    if (NULL == *mm_desc) {
        status = UCS_ERR_NO_RESOURCE;
        goto err_mem;
    }

    apid = xpmem_get(segid, XPMEM_RDWR, XPMEM_PERMIT_MODE, NULL);
    if (apid < 0) {
        ucs_error("Failed to xpmem_get: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_xget;
    }

    addr.apid = apid;
    addr.offset = diff;

    (*mm_desc)->address = xpmem_attach(addr, length, NULL);
    if ((*mm_desc)->address < 0) {
        ucs_error("Failed to xpmem_attach: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_xattach;
    }
    (*mm_desc)->cookie = apid;

    return UCS_OK;

err_xattach:
    xpmem_release(apid);
err_xget:
    ucs_free(*mm_desc);
err_mem:
    return status;
}

static ucs_status_t uct_xpmem_detach(uct_mm_mapped_desc_t *mm_desc)
{
    int rc = xpmem_detach(mm_desc->address);
    if (rc < 0) {
        ucs_error("Failed to xpmem_detach: %m");
        return UCS_ERR_IO_ERROR;
    }
    rc = xpmem_release((xpmem_apid_t) mm_desc->cookie);
    if (rc < 0) {
        ucs_error("Failed to xpmem_release: %m");
        return UCS_ERR_IO_ERROR;
    }
    ucs_free(mm_desc);
    return UCS_OK;
}

static ucs_status_t uct_xpmem_alloc(size_t *length_p, ucs_ternary_value_t
                                    hugetlb, void **address_p, uct_mm_id_t
                                    *mmid_p UCS_MEMTRACK_ARG)
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

    return uct_xmpem_reg(*address_p, length_aligned, mmid_p);
err:
    return status;
}

static ucs_status_t uct_xpmem_free(void *address, uct_mm_id_t mm_id)
{
    ucs_status_t status = uct_xpmem_dereg(mm_id);
    if (UCS_OK != status) {
        return status;
    }

    ucs_free(address);
    return UCS_OK;
}

static uct_mm_mapper_ops_t uct_xpmem_mapper_ops = {
    .query   = uct_xpmem_query,
    .reg     = uct_xmpem_reg,
    .dereg   = uct_xpmem_dereg,
    .alloc   = uct_xpmem_alloc,
    .attach  = uct_xpmem_attach,
    .detach  = uct_xpmem_detach,
    .free    = uct_xpmem_free
};

UCT_MM_COMPONENT_DEFINE(uct_xpmem_pd, "xpmem", &uct_xpmem_mapper_ops)
UCT_PD_REGISTER_TL(&uct_xpmem_pd, &uct_mm_tl);
