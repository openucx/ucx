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
#include <ucs/sys/sys.h>

#define UCT_MM_SYSV_PERM (S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP)
#define UCT_MM_SYSV_MSTR (UCT_MM_SYSV_PERM | IPC_CREAT | IPC_EXCL)

static ucs_status_t
uct_sysv_alloc(size_t *length_p, ucs_ternary_value_t hugetlb,
               void **address_p, uct_mm_id_t *mmid_p UCS_MEMTRACK_ARG)
{
    ucs_status_t status = UCS_ERR_NO_MEMORY;
    int flags, shmid = 0;

    flags = UCT_MM_SYSV_MSTR;

    if (0 == *length_p) {
        ucs_error("Unexpected length %zu", *length_p);
        status = UCS_ERR_INVALID_PARAM;
        goto err;
    }

    if (hugetlb != UCS_NO) {
        status = ucs_sysv_alloc(length_p, address_p, flags | SHM_HUGETLB, &shmid
                                UCS_MEMTRACK_VAL);
        if (status == UCS_OK) {
            goto out_ok;
        }

        ucs_debug("mm failed to allocate %zu bytes with hugetlb", *length_p);
    }

    if (hugetlb != UCS_YES) {
        status = ucs_sysv_alloc(length_p, address_p, flags , &shmid UCS_MEMTRACK_VAL);
        if (status == UCS_OK) {
            goto out_ok;
        }

        ucs_debug("mm failed to allocate %zu bytes without hugetlb", *length_p);
    }

err:
    return status;

out_ok:
    *mmid_p = shmid;
    return UCS_OK;
}

static ucs_status_t uct_sysv_attach(uct_mm_id_t mmid, size_t length, void **address_p)
{
    void *ptr;

    ptr = shmat(mmid, NULL, 0);
    if (ptr == MAP_FAILED) {
        ucs_error("shmat(shmid=%d) failed: %m", (int)mmid);
        return UCS_ERR_SHMEM_SEGMENT;
    }

    *address_p = ptr;
    return UCS_OK;
}

static uct_mm_mapper_ops_t uct_sysv_mapper_ops = {
   .query   = ucs_empty_function_return_success,
   .reg     = NULL,
   .dereg   = NULL,
   .alloc   = uct_sysv_alloc,
   .attach  = uct_sysv_attach,
   .detach  = ucs_sysv_free,
   .free    = ucs_sysv_free
};

UCT_MM_COMPONENT_DEFINE(uct_sysv_pd, "sysv", &uct_sysv_mapper_ops)
UCT_PD_REGISTER_TL(&uct_sysv_pd, &uct_mm_tl);
