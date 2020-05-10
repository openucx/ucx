/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <uct/sm/mm/base/mm_md.h>
#include <uct/sm/mm/base/mm_iface.h>
#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>
#include <ucs/sys/sys.h>


#define UCT_MM_SYSV_PERM (S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP)
#define UCT_MM_SYSV_MSTR (UCT_MM_SYSV_PERM | IPC_CREAT | IPC_EXCL)

typedef struct uct_sysv_packed_rkey {
    uint32_t                shmid;
    uintptr_t               owner_ptr;
} UCS_S_PACKED uct_sysv_packed_rkey_t;

typedef struct uct_sysv_md_config {
    uct_mm_md_config_t      super;
} uct_sysv_md_config_t;

static ucs_config_field_t uct_sysv_md_config_table[] = {
  {"MM_", "", NULL,
   ucs_offsetof(uct_sysv_md_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_mm_md_config_table)},

  {NULL}
};

static ucs_status_t uct_sysv_md_query(uct_md_h md, uct_md_attr_t *md_attr)
{
    uct_mm_md_query(md, md_attr, 1);
    md_attr->rkey_packed_size = sizeof(uct_sysv_packed_rkey_t);
    return UCS_OK;
}

static ucs_status_t uct_sysv_mem_attach_common(int shmid, void **address_p)
{
    void *address;

    address = shmat(shmid, NULL, 0);
    if (address == MAP_FAILED) {
        ucs_error("shmat(shmid=%d) failed: %m", shmid);
        *address_p = NULL; /* GCC 8.3.1 reports error without it */
        return UCS_ERR_SHMEM_SEGMENT;
    }

    *address_p = address;
    ucs_trace("attached remote segment %d at address %p", (int)shmid, address);
    return UCS_OK;
}

static ucs_status_t
uct_sysv_mem_alloc(uct_md_h tl_md, size_t *length_p, void **address_p,
                   unsigned flags, const char *alloc_name, uct_mem_h *memh_p)
{
    uct_mm_md_t *md = ucs_derived_of(tl_md, uct_mm_md_t);
    ucs_status_t status;
    uct_mm_seg_t *seg;
    int shmid;

    status = uct_mm_seg_new(*address_p, *length_p, &seg);
    if (status != UCS_OK) {
        return status;
    }

#ifdef SHM_HUGETLB
    if (md->config->hugetlb_mode != UCS_NO) {
        status = ucs_sysv_alloc(&seg->length, seg->length * 2, &seg->address,
                                UCT_MM_SYSV_MSTR | SHM_HUGETLB, alloc_name,
                                &shmid);
        if (status == UCS_OK) {
            goto out_ok;
        }

        ucs_debug("mm failed to allocate %zu bytes with hugetlb", seg->length);
    }
#else
    status = UCS_ERR_UNSUPPORTED;
#endif

    if (md->config->hugetlb_mode != UCS_YES) {
        status = ucs_sysv_alloc(&seg->length, SIZE_MAX, &seg->address,
                                UCT_MM_SYSV_MSTR, alloc_name, &shmid);
        if (status == UCS_OK) {
            goto out_ok;
        }

        ucs_debug("mm failed to allocate %zu bytes without hugetlb", seg->length);
    }

    ucs_error("failed to allocate %zu bytes with mm for %s", seg->length,
              alloc_name);
    ucs_free(seg);
    return status;

out_ok:
    seg->seg_id = shmid;
    *address_p  = seg->address;
    *length_p   = seg->length;
    *memh_p     = seg;
    return UCS_OK;
}

static ucs_status_t uct_sysv_mem_free(uct_md_h tl_md, uct_mem_h memh)
{
    uct_mm_seg_t *seg = memh;
    ucs_status_t status;

    status = ucs_sysv_free(seg->address);
    if (status != UCS_OK) {
        return status;
    }

    ucs_free(seg);
    return UCS_OK;
}

static ucs_status_t
uct_sysv_md_mkey_pack(uct_md_h md, uct_mem_h memh, void *rkey_buffer)
{
    uct_sysv_packed_rkey_t *packed_rkey = rkey_buffer;
    const uct_mm_seg_t     *seg         = memh;

    packed_rkey->shmid     = seg->seg_id;
    packed_rkey->owner_ptr = (uintptr_t)seg->address;
    return UCS_OK;
}

static ucs_status_t uct_sysv_mem_attach(uct_mm_md_t *md, uct_mm_seg_id_t seg_id,
                                        size_t length, const void *iface_addr,
                                        uct_mm_remote_seg_t *rseg)
{
    return uct_sysv_mem_attach_common(seg_id, &rseg->address);
}

static void uct_sysv_mem_detach(uct_mm_md_t *md, const uct_mm_remote_seg_t *rseg)
{
    ucs_sysv_free(rseg->address);
}

static ucs_status_t
uct_sysv_rkey_unpack(uct_component_t *component, const void *rkey_buffer,
                     uct_rkey_t *rkey_p, void **handle_p)
{
    const uct_sysv_packed_rkey_t *packed_rkey = rkey_buffer;
    ucs_status_t status;
    void *address;

    status = uct_sysv_mem_attach_common(packed_rkey->shmid, &address);
    if (status != UCS_OK) {
        return status;
    }

    *handle_p = address;
    uct_mm_md_make_rkey(address, packed_rkey->owner_ptr, rkey_p);
    return UCS_OK;
}

static ucs_status_t
uct_sysv_rkey_release(uct_component_t *component, uct_rkey_t rkey, void *handle)
{
    return ucs_sysv_free(handle);
}

static uct_mm_md_mapper_ops_t uct_sysv_md_ops = {
   .super = {
        .close                  = uct_mm_md_close,
        .query                  = uct_sysv_md_query,
        .mem_alloc              = uct_sysv_mem_alloc,
        .mem_free               = uct_sysv_mem_free,
        .mem_advise             = (uct_md_mem_advise_func_t)ucs_empty_function_return_unsupported,
        .mem_reg                = (uct_md_mem_reg_func_t)ucs_empty_function_return_unsupported,
        .mem_dereg              = (uct_md_mem_dereg_func_t)ucs_empty_function_return_unsupported,
        .mkey_pack              = uct_sysv_md_mkey_pack,
        .is_sockaddr_accessible = (uct_md_is_sockaddr_accessible_func_t)ucs_empty_function_return_zero,
        .detect_memory_type     = (uct_md_detect_memory_type_func_t)ucs_empty_function_return_unsupported
    },
   .query                       = (uct_mm_mapper_query_func_t)
                                      ucs_empty_function_return_success,
   .iface_addr_length           = (uct_mm_mapper_iface_addr_length_func_t)
                                      ucs_empty_function_return_zero_int64,
   .iface_addr_pack             = (uct_mm_mapper_iface_addr_pack_func_t)
                                      ucs_empty_function_return_success,
   .mem_attach                  = uct_sysv_mem_attach,
   .mem_detach                  = uct_sysv_mem_detach,
   .is_reachable                = (uct_mm_mapper_is_reachable_func_t)ucs_empty_function_return_one
};

UCT_MM_TL_DEFINE(sysv, &uct_sysv_md_ops, uct_sysv_rkey_unpack,
                 uct_sysv_rkey_release, "SYSV_")
