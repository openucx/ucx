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
    md_attr->rkey_packed_size = sizeof(uct_mm_packed_rkey_t);
    return UCS_OK;
}

static ucs_status_t
uct_sysv_alloc(uct_md_h md, size_t *length_p, ucs_ternary_value_t hugetlb,
               unsigned md_map_flags, const char *alloc_name, void **address_p,
               uct_mm_id_t *mmid_p, const char **path_p)
{
    ucs_status_t status = UCS_ERR_NO_MEMORY;
    int flags, shmid = 0;

    flags = UCT_MM_SYSV_MSTR;

    if (0 == *length_p) {
        ucs_error("Unexpected length %zu", *length_p);
        status = UCS_ERR_INVALID_PARAM;
        goto err;
    }

    if (!(md_map_flags & UCT_MD_MEM_FLAG_FIXED)) {
        *address_p = NULL;
    }

#ifdef SHM_HUGETLB
    if (hugetlb != UCS_NO) {
        status = ucs_sysv_alloc(length_p, (*length_p) * 2, address_p,
                                flags | SHM_HUGETLB, alloc_name, &shmid);
        if (status == UCS_OK) {
            goto out_ok;
        }

        ucs_debug("mm failed to allocate %zu bytes with hugetlb", *length_p);
    }
#endif

    if (hugetlb != UCS_YES) {
        status = ucs_sysv_alloc(length_p, SIZE_MAX, address_p, flags, alloc_name,
                                &shmid);
        if (status == UCS_OK) {
            goto out_ok;
        }

        ucs_debug("mm failed to allocate %zu bytes without hugetlb", *length_p);
    }

err:
    ucs_error("failed to allocate %zu bytes with mm for %s", *length_p, alloc_name);
    return status;

out_ok:
    *mmid_p = shmid;
    return UCS_OK;
}

static ucs_status_t uct_sysv_attach(uct_mm_id_t mmid, size_t length,
                                    void *remote_address,
                                    void **local_address,
                                    uint64_t *cookie, const char *path)
{
    void *ptr;

    ptr = shmat(mmid, NULL, 0);
    if (ptr == MAP_FAILED) {
        ucs_error("shmat(shmid=%d) failed: %m", (int)mmid);
        return UCS_ERR_SHMEM_SEGMENT;
    }

    ucs_trace("attached remote segment %d remote_address %p at address %p",
              (int)mmid, remote_address, ptr);
    *local_address = ptr;
    *cookie = 0xdeadbeef;

    return UCS_OK;
}

static ucs_status_t uct_sysv_detach(uct_mm_remote_seg_t *mm_desc)
{
    ucs_status_t status = ucs_sysv_free(mm_desc->address);
    if (UCS_OK != status) {
        return status;
    }

    return UCS_OK;
}

static ucs_status_t uct_sysv_free(void *address, uct_mm_id_t mm_id, size_t length,
                                  const char *path)
{
    return ucs_sysv_free(address);
}

static uint8_t uct_sysv_get_priority()
{
    return 0;
}

static uct_mm_md_mapper_ops_t uct_sysv_md_ops = {
    .super = {
        .close                  = uct_mm_md_close,
        .query                  = uct_sysv_md_query,
        .mem_alloc              = uct_mm_mem_alloc,
        .mem_free               = uct_mm_mem_free,
        .mem_advise             = (uct_md_mem_advise_func_t)ucs_empty_function_return_unsupported,
        .mem_reg                = (uct_md_mem_reg_func_t)ucs_empty_function_return_unsupported,
        .mem_dereg              = (uct_md_mem_dereg_func_t)ucs_empty_function_return_unsupported,
        .mkey_pack              = uct_mm_mkey_pack,
        .is_sockaddr_accessible = (uct_md_is_sockaddr_accessible_func_t)ucs_empty_function_return_zero,
        .detect_memory_type     = (uct_md_detect_memory_type_func_t)ucs_empty_function_return_unsupported
    },
    .query                      = ucs_empty_function_return_success,
    .get_priority               = uct_sysv_get_priority,
    .reg                        = NULL,
    .dereg                      = NULL,
    .alloc                      = uct_sysv_alloc,
    .attach                     = uct_sysv_attach,
    .detach                     = uct_sysv_detach,
    .free                       = uct_sysv_free
};

UCT_MM_TL_DEFINE(sysv, &uct_sysv_md_ops, uct_mm_rkey_unpack,
                 uct_mm_rkey_release, "SYSV_")
