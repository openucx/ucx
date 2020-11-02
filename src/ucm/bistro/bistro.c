/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.       ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <dlfcn.h>
#include <stdlib.h>

#include <ucm/bistro/bistro.h>
#include <ucm/bistro/bistro_int.h>

ucs_status_t ucm_bistro_remove_restore_point(ucm_bistro_restore_point_t *rp)
{
    ucs_assert(rp != NULL);
    free(rp);
    return UCS_OK;
}

static void *ucm_bistro_page_align_ptr(void *ptr)
{
    return (void*)ucs_align_down((uintptr_t)ptr, ucm_get_page_size());
}

static ucs_status_t ucm_bistro_protect(void *addr, size_t len, int prot)
{
    void *aligned = ucm_bistro_page_align_ptr(addr);
    size_t size   = UCS_PTR_BYTE_DIFF(aligned, addr) + len;
    int res;

    res = mprotect(aligned, size, prot) ? UCS_ERR_INVALID_PARAM : UCS_OK;
    if (res) {
        ucm_error("Failed to change page protection: %m");
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

ucs_status_t ucm_bistro_apply_patch(void *dst, void *patch, size_t len)
{
    ucs_status_t status;

    status = ucm_bistro_protect(dst, len, UCM_PROT_READ_WRITE_EXEC);
    if (UCS_STATUS_IS_ERR(status)) {
        return status;
    }

    memcpy(dst, patch, len);

    status = ucm_bistro_protect(dst, len, UCM_PROT_READ_EXEC);
    if (!UCS_STATUS_IS_ERR(status)) {
        ucs_clear_cache(dst, UCS_PTR_BYTE_OFFSET(dst, len));
    }
    return status;
}

#if defined(__x86_64__) || defined (__aarch64__)
struct ucm_bistro_restore_point {
    void               *addr;     /* address of function to restore */
    size_t             patch_len; /* patch length */
    char               orig[0];   /* orig func code */
};

ucs_status_t ucm_bistro_create_restore_point(void *addr, size_t len,
                                             ucm_bistro_restore_point_t **rp)
{
    ucm_bistro_restore_point_t *point;

    if (rp == NULL) {
        /* restore point is not required */
        return UCS_OK;
    }

    point = malloc(sizeof(*point) + len);
    if (point == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    *rp              = point;
    point->addr      = addr;
    point->patch_len = len;
    memcpy(point->orig, addr, len);

    return UCS_OK;
}

ucs_status_t ucm_bistro_restore(ucm_bistro_restore_point_t *rp)
{
    ucs_status_t status;

    status = ucm_bistro_apply_patch(rp->addr, rp->orig, rp->patch_len);
    if (!UCS_STATUS_IS_ERR(status)) {
        ucm_bistro_remove_restore_point(rp);
    }

    return status;
}

void *ucm_bistro_restore_addr(ucm_bistro_restore_point_t *rp)
{
    ucs_assert(rp != NULL);
    return rp->addr;
}

#endif
