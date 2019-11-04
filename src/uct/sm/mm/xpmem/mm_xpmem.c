/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * Copyright (c) Los Alamos National Security, LLC. 2016.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "xpmem.h"

#include <uct/sm/mm/base/mm_md.h>
#include <uct/sm/mm/base/mm_iface.h>
#include <ucs/debug/memtrack.h>
#include <ucs/type/init_once.h>
#include <ucs/debug/log.h>


typedef struct uct_xpmem_md_config {
    uct_mm_md_config_t      super;
} uct_xpmem_md_config_t;

typedef struct uct_xpmem_iface_addr {
    xpmem_segid_t           xsegid;
} UCS_S_PACKED uct_xpmem_iface_addr_t;

typedef struct uct_xpmem_packed_rkey {
    xpmem_segid_t           xsegid;
    uintptr_t               address;
    size_t                  length;
} UCS_S_PACKED uct_xpmem_packed_rkey_t;

static ucs_config_field_t uct_xpmem_md_config_table[] = {
  {"MM_", "", NULL,
   ucs_offsetof(uct_xpmem_md_config_t, super),
   UCS_CONFIG_TYPE_TABLE(uct_mm_md_config_table)},

  {NULL}
};

static ucs_status_t uct_xpmem_query()
{
    int version;

    version = xpmem_version();
    if (version < 0) {
        ucs_debug("xpmem_version() returned %d (%m), xpmem is unavailable",
                  version);
        return UCS_ERR_UNSUPPORTED;
    }

    ucs_debug("xpmem version: %d", version);
    return UCS_OK;
}

static ucs_status_t uct_xpmem_md_query(uct_md_h md, uct_md_attr_t *md_attr)
{
    uct_mm_md_query(md, md_attr, 0);

    md_attr->cap.flags         |= UCT_MD_FLAG_REG;
    md_attr->reg_cost.overhead  = 5.0e-9;
    md_attr->reg_cost.growth    = 0;
    md_attr->cap.max_reg        = ULONG_MAX;
    md_attr->cap.reg_mem_types  = UCS_BIT(UCS_MEMORY_TYPE_HOST);
    md_attr->rkey_packed_size   = sizeof(uct_xpmem_packed_rkey_t);

    return UCS_OK;
}

static ucs_status_t uct_xpmem_get_global_xsegid(xpmem_segid_t *xsegid_p)
{
    static ucs_init_once_t init_once = UCS_INIT_ONCE_INITIALIZER;
    static xpmem_segid_t global_xsegid = -1;

    if (ucs_unlikely(global_xsegid == -1)) {
        /* double-checked locking */
        UCS_INIT_ONCE(&init_once) {
            if (global_xsegid == -1) {
                global_xsegid = xpmem_make(0, XPMEM_MAXADDR_SIZE,
                                           XPMEM_PERMIT_MODE, (void*)0600);
                VALGRIND_MAKE_MEM_DEFINED(&global_xsegid, sizeof(global_xsegid));
            }
        }

        if (global_xsegid < 0) {
            ucs_error("failed to register address space xpmem: %m");
            return UCS_ERR_IO_ERROR;
        }

        ucs_debug("xpmem registered global segment id 0x%llx", global_xsegid);
    }

    *xsegid_p = global_xsegid;
    return UCS_OK;
}

static ucs_status_t uct_xmpem_mem_reg(uct_md_h md, void *address, size_t length,
                                      unsigned flags, uct_mem_h *memh_p)
{
    ucs_status_t status;
    uct_mm_seg_t *seg;

    status = uct_mm_seg_new(address, length, &seg);
    if (status != UCS_OK) {
        return status;
    }

    seg->seg_id  = (uintptr_t)address; /* to be used by mem_attach */
    *memh_p      = seg;
    return UCS_OK;
}

static ucs_status_t uct_xmpem_mem_dereg(uct_md_h md, uct_mem_h memh)
{
    uct_mm_seg_t *seg = memh;
    ucs_free(seg);
    return UCS_OK;
}

static ucs_status_t
uct_xpmem_mem_attach_common(xpmem_segid_t xsegid, uintptr_t remote_address,
                            size_t length, uct_mm_remote_seg_t *rseg)
{
    struct xpmem_addr addr;
    uintptr_t start, end;
    ucs_status_t status;
    void *address;

    start = ucs_align_down_pow2(remote_address,          ucs_get_page_size());
    end   = ucs_align_up_pow2  (remote_address + length, ucs_get_page_size());

    addr.offset = start;
    addr.apid   = xpmem_get(xsegid, XPMEM_RDWR, XPMEM_PERMIT_MODE, NULL);
    VALGRIND_MAKE_MEM_DEFINED(&addr.apid, sizeof(addr.apid));
    if (addr.apid < 0) {
        ucs_error("Failed to acquire xpmem segment 0x%llx: %m", xsegid);
        status = UCS_ERR_IO_ERROR;
        goto err_xget;
    }

    ucs_trace("xpmem acquired segment 0x%llx apid 0x%llx remote_address %p",
              xsegid, addr.apid, (void*)remote_address);

    address = xpmem_attach(addr, end - start, NULL);
    VALGRIND_MAKE_MEM_DEFINED(&address, sizeof(address));
    if (address == MAP_FAILED) {
        ucs_error("Failed to attach xpmem segment 0x%llx apid 0x%llx "
                  "with length %zu: %m", xsegid, addr.apid, length);
        status = UCS_ERR_IO_ERROR;
        goto err_xattach;
    }

    rseg->address = UCS_PTR_BYTE_OFFSET(address, remote_address - start);
    rseg->cookie  = (void*)addr.apid;
    VALGRIND_MAKE_MEM_DEFINED(rseg->address, length);

    ucs_trace("xpmem attached segment 0x%llx apid 0x%llx 0x%lx..0x%lx at %p",
              xsegid, addr.apid, start, end, address);
    return UCS_OK;

err_xattach:
    xpmem_release(addr.apid);
err_xget:
    return status;
}

static void uct_xpmem_mem_detach_common(const uct_mm_remote_seg_t *rseg)
{
    xpmem_apid_t apid = (uintptr_t)rseg->cookie;
    void *address;
    int ret;

    address = ucs_align_down_pow2_ptr(rseg->address, ucs_get_page_size());

    ucs_trace("xpmem detaching address %p", address);
    ret = xpmem_detach(address);
    if (ret < 0) {
        ucs_error("Failed to xpmem_detach: %m");
        return;
    }

    ucs_trace("xpmem releasing segment apid 0x%llx", apid);
    ret = xpmem_release(apid);
    if (ret < 0) {
        ucs_error("Failed to release xpmem segment apid 0x%llx", apid);
        return;
    }
}

static ucs_status_t
uct_xpmem_mkey_pack(uct_md_h md, uct_mem_h memh, void *rkey_buffer)
{
    uct_mm_seg_t                    *seg = memh;
    uct_xpmem_packed_rkey_t *packed_rkey = rkey_buffer;
    ucs_status_t status;

    ucs_assert((uintptr_t)seg->address == seg->seg_id); /* sanity */

    status = uct_xpmem_get_global_xsegid(&packed_rkey->xsegid);
    if (status != UCS_OK) {
        return status;
    }

    packed_rkey->address = (uintptr_t)seg->address;
    packed_rkey->length  = seg->length;
    return UCS_OK;
}

static size_t uct_xpmem_iface_addr_length(uct_mm_md_t *md)
{
    return sizeof(uct_xpmem_iface_addr_t);
}

static ucs_status_t uct_xpmem_iface_addr_pack(uct_mm_md_t *md, void *buffer)
{
    uct_xpmem_iface_addr_t *xpmem_iface_addr = buffer;

    return uct_xpmem_get_global_xsegid(&xpmem_iface_addr->xsegid);
}

static ucs_status_t uct_xpmem_mem_attach(uct_mm_md_t *md, uct_mm_seg_id_t seg_id,
                                         size_t length, const void *iface_addr,
                                         uct_mm_remote_seg_t *rseg)
{
    const uct_xpmem_iface_addr_t *xpmem_iface_addr = iface_addr;

    ucs_assert(xpmem_iface_addr != NULL);
    return uct_xpmem_mem_attach_common(xpmem_iface_addr->xsegid,
                                       seg_id /* remote_address */, length, rseg);
}

static void uct_xpmem_mem_detach(uct_mm_md_t *md,
                                 const uct_mm_remote_seg_t *rseg)
{
    uct_xpmem_mem_detach_common(rseg);
}

static ucs_status_t
uct_xpmem_rkey_unpack(uct_component_t *component, const void *rkey_buffer,
                      uct_rkey_t *rkey_p, void **handle_p)
{
    const uct_xpmem_packed_rkey_t *packed_rkey = rkey_buffer;
    uct_mm_remote_seg_t *rseg;
    ucs_status_t status;

    rseg = ucs_malloc(sizeof(*rseg), "xpmem_rseg");
    if (rseg == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    status = uct_xpmem_mem_attach_common(packed_rkey->xsegid,
                                         packed_rkey->address,
                                         packed_rkey->length,
                                         rseg);
    if (status != UCS_OK) {
        ucs_free(rseg);
        return status;
    }

    uct_mm_md_make_rkey(rseg->address, packed_rkey->address, rkey_p);
    *handle_p = rseg;

    return UCS_OK;
}

static ucs_status_t
uct_xpmem_rkey_release(uct_component_t *component, uct_rkey_t rkey, void *handle)
{
    uct_mm_remote_seg_t *rseg = handle;

    uct_xpmem_mem_detach_common(rseg);
    ucs_free(rseg);
    return UCS_OK;
}

static uct_mm_md_mapper_ops_t uct_xpmem_md_ops = {
    .super = {
        .close                  = uct_mm_md_close,
        .query                  = uct_xpmem_md_query,
        .mem_alloc              = (uct_md_mem_alloc_func_t)ucs_empty_function_return_unsupported,
        .mem_free               = (uct_md_mem_free_func_t)ucs_empty_function_return_unsupported,
        .mem_advise             = (uct_md_mem_advise_func_t)ucs_empty_function_return_unsupported,
        .mem_reg                = uct_xmpem_mem_reg,
        .mem_dereg              = uct_xmpem_mem_dereg,
        .mkey_pack              = uct_xpmem_mkey_pack,
        .is_sockaddr_accessible = (uct_md_is_sockaddr_accessible_func_t)ucs_empty_function_return_zero,
        .detect_memory_type     = (uct_md_detect_memory_type_func_t)ucs_empty_function_return_unsupported
    },
   .query                       = uct_xpmem_query,
   .iface_addr_length           = uct_xpmem_iface_addr_length,
   .iface_addr_pack             = uct_xpmem_iface_addr_pack,
   .mem_attach                  = uct_xpmem_mem_attach,
   .mem_detach                  = uct_xpmem_mem_detach
};

UCT_MM_TL_DEFINE(xpmem, &uct_xpmem_md_ops, uct_xpmem_rkey_unpack,
                 uct_xpmem_rkey_release, "XPMEM_")
