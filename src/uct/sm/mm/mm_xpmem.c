/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * Copyright (c) Los Alamos National Security, LLC. 2016.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "mm_md.h"
#include "mm_iface.h"

#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>
#include "xpmem.h"


static ucs_status_t uct_xpmem_query()
{
    int version;

    version = xpmem_version();
    if (version < 0) {
        ucs_debug("Failed to query XPMEM version %d, %m", version);
        return UCS_ERR_UNSUPPORTED;
    }
    return UCS_OK;
}

static size_t uct_xpmem_get_path_size(uct_md_h md)
{
    return 0;
}

static uint8_t uct_xpmem_get_priority()
{
    return 0;
}

static ucs_status_t uct_xmpem_reg(void *address, size_t size, uct_mm_id_t *mmid_p)
{
    xpmem_segid_t segid;
    void *start, *end;

    start = ucs_align_down_pow2_ptr(address, ucs_get_page_size());
    end   = ucs_align_up_pow2_ptr(address + size, ucs_get_page_size());
    ucs_assert_always(start <= end);

    segid = xpmem_make(start, end - start, XPMEM_PERMIT_MODE, (void*)0666);
    VALGRIND_MAKE_MEM_DEFINED(&segid, sizeof(segid));
    if (segid < 0) {
        ucs_error("Failed to register %p..%p with xpmem: %m",
                  start, end);
        return UCS_ERR_IO_ERROR;
    }

    ucs_trace("xpmem registered %p..%p segment 0x%llx", start, end, segid);
    *mmid_p = segid;
    return UCS_OK;
}

static ucs_status_t uct_xpmem_dereg(uct_mm_id_t mmid)
{
    int ret;

    ret = xpmem_remove(mmid);
    if (ret < 0) {
        /* No error since there a chance that it already was released
         * or deregistered */
        ucs_debug("Failed to remove xpmem segment 0x%"PRIx64": %m", mmid);
    }

    ucs_trace("xpmem removed segment 0x%"PRIx64, mmid);
    return UCS_OK;
}

static ucs_status_t uct_xpmem_attach(uct_mm_id_t mmid, size_t length, 
                                     void *remote_address, void **local_address,
                                     uint64_t *cookie, const char *path)
{
    struct xpmem_addr addr;
    ucs_status_t status;
    ptrdiff_t offset;
    void *address;

    addr.offset = 0;
    addr.apid   = xpmem_get(mmid, XPMEM_RDWR, XPMEM_PERMIT_MODE, NULL);
    VALGRIND_MAKE_MEM_DEFINED(&addr.apid, sizeof(addr.apid));
    if (addr.apid < 0) {
        ucs_error("Failed to acquire xpmem segment 0x%"PRIx64": %m", mmid);
        status = UCS_ERR_IO_ERROR;
        goto err_xget;
    }

    ucs_trace("xpmem acquired segment 0x%"PRIx64" apid 0x%llx remote_address %p",
              mmid, addr.apid, remote_address);

    offset  = ((uintptr_t)remote_address) % ucs_get_page_size();
    address = xpmem_attach(addr, length + offset, NULL);
    VALGRIND_MAKE_MEM_DEFINED(&address, sizeof(address));
    if (address == MAP_FAILED) {
        ucs_error("Failed to attach xpmem segment 0x%"PRIx64" apid 0x%llx "
                  "with length %zu: %m", mmid, addr.apid, length);
        status = UCS_ERR_IO_ERROR;
        goto err_xattach;
    }

    VALGRIND_MAKE_MEM_DEFINED(address + offset, length);

    *local_address = address + offset;
    *cookie        = addr.apid;

    ucs_trace("xpmem attached segment 0x%"PRIx64" apid 0x%llx %p..%p at %p (+%zd)",
              mmid, addr.apid, remote_address, remote_address + length, address, offset);
    return UCS_OK;

err_xattach:
    xpmem_release(addr.apid);
err_xget:
    return status;
}

static ucs_status_t uct_xpmem_detach(uct_mm_remote_seg_t *mm_desc)
{
    xpmem_apid_t apid = mm_desc->cookie;
    void *address;
    int ret;

    address = ucs_align_down_pow2_ptr(mm_desc->address, ucs_get_page_size());

    ucs_trace("xpmem detaching address %p", address);
    ret = xpmem_detach(address);
    if (ret < 0) {
        ucs_error("Failed to xpmem_detach: %m");
        return UCS_ERR_IO_ERROR;
    }

    VALGRIND_MAKE_MEM_UNDEFINED(mm_desc->address, mm_desc->length);

    ucs_trace("xpmem releasing segment apid 0x%llx", apid);
    ret = xpmem_release(apid);
    if (ret < 0) {
        ucs_error("Failed to release xpmem segment apid 0x%llx", apid);
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

static ucs_status_t uct_xpmem_alloc(uct_md_h md, size_t *length_p,
                                    ucs_ternary_value_t hugetlb,
                                    unsigned md_map_flags, void **address_p,
                                    uct_mm_id_t *mmid_p, const char **path_p
                                    UCS_MEMTRACK_ARG)
{
    ucs_status_t status;
    int mmap_flags;

    if (0 == *length_p) {
        ucs_error("Unexpected length %zu", *length_p);
        status = UCS_ERR_INVALID_PARAM;
        goto out;
    }

    if (md_map_flags & UCT_MD_MEM_FLAG_FIXED) {
        mmap_flags = MAP_FIXED;
    } else {
        *address_p = NULL;
        mmap_flags = 0;
    }

    /* TBD: any ideas for better allocation */
    status = ucs_mmap_alloc(length_p, address_p, mmap_flags UCS_MEMTRACK_VAL);
    if (status != UCS_OK) {
        ucs_error("Failed to allocate %zu bytes of memory", *length_p);
        goto out;
    }

    ucs_trace("xpmem allocated address %p length %zu", *address_p, *length_p);

    status = uct_xmpem_reg(*address_p, *length_p, mmid_p);
    if (UCS_OK != status) {
        ucs_free(*address_p);
        goto out;
    }

    VALGRIND_MAKE_MEM_DEFINED(*address_p, *length_p);
    status     = UCS_OK;

out:
    return status;
}

static ucs_status_t uct_xpmem_free(void *address, uct_mm_id_t mmid, size_t length,
                                   const char *path)
{
    ucs_status_t status;

    status = uct_xpmem_dereg(mmid);
    if (UCS_OK != status) {
        return status;
    }

    return ucs_mmap_free(address, length);
}

static uct_mm_mapper_ops_t uct_xpmem_mapper_ops = {
    .query   = uct_xpmem_query,
    .get_path_size = uct_xpmem_get_path_size,
    .get_priority = uct_xpmem_get_priority,
    .reg     = uct_xmpem_reg,
    .dereg   = uct_xpmem_dereg,
    .alloc   = uct_xpmem_alloc,
    .attach  = uct_xpmem_attach,
    .detach  = uct_xpmem_detach,
    .free    = uct_xpmem_free
};

UCT_MM_COMPONENT_DEFINE(uct_xpmem_md, "xpmem", &uct_xpmem_mapper_ops, uct, "XPMEM_")
UCT_MD_REGISTER_TL(&uct_xpmem_md, &uct_mm_tl);
