/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2018. ALL RIGHTS RESERVED.
 * Copyright (C) Tactical Computing Labs, LLC. 2022. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <dlfcn.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

#include <ucm/bistro/bistro.h>
#include <ucm/bistro/bistro_int.h>
#include <ucs/type/serialize.h>
#include <ucs/sys/ptr_arith.h>
#include <ucs/time/time.h>
#include <ucs/arch/atomic.h>


ucs_status_t ucm_bistro_remove_restore_point(ucm_bistro_restore_point_t *rp)
{
    ucm_assert(rp != NULL);
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

void ucm_bistro_modify_code(void *dst, const ucm_bistro_lock_t *bytes)
{
    uint16_t value16;
    uint32_t value32;

    UCS_STATIC_ASSERT((sizeof(*bytes) == sizeof(value16)) ||
                      (sizeof(*bytes) == sizeof(value32)) ||
                      (sizeof(*bytes) == 0));

    if (sizeof(*bytes) == sizeof(value16)) {
        memcpy(&value16, bytes, sizeof(value16));
        (void)ucs_atomic_swap16(dst, value16);
    } else if (sizeof(*bytes) == sizeof(value32)) {
        memcpy(&value32, bytes, sizeof(value32));
        (void)ucs_atomic_swap32(dst, value32);
    }
}

ucs_status_t
ucm_bistro_apply_patch_atomic(void *dst, const void *patch, size_t len)
{
    size_t skip           = sizeof(ucm_bistro_lock_t);
    double grace_duration = 5e-3;
    double deadline;
    ucs_status_t status;

    status = ucm_bistro_protect(dst, len, UCM_PROT_READ_WRITE_EXEC);
    if (UCS_STATUS_IS_ERR(status)) {
        return status;
    }

    /* Lock the codepatch and wait for existing flows to complete */
    ucm_bistro_patch_lock(dst);
    ucs_clear_cache(dst, UCS_PTR_BYTE_OFFSET(dst, len));

    deadline = ucm_get_time() + grace_duration;
    while (ucm_get_time() < deadline) {
        sched_yield();
    }

    /* Copy the payload behind the lock */
    memcpy(UCS_PTR_BYTE_OFFSET(dst, skip), UCS_PTR_BYTE_OFFSET(patch, skip),
           len - skip);
    ucs_clear_cache(dst, UCS_PTR_BYTE_OFFSET(dst, len));

    /* Unlock the codepath */
    ucm_bistro_modify_code(dst, patch);

    status = ucm_bistro_protect(dst, len, UCM_PROT_READ_EXEC);
    ucs_clear_cache(dst, UCS_PTR_BYTE_OFFSET(dst, len));

    return status;
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

#if defined(__x86_64__) || defined (__aarch64__) || defined (__riscv)
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
    ucm_assert(rp != NULL);
    return rp->addr;
}

void *ucm_bistro_allocate_code(size_t size)
{
    static const size_t mmap_size = 16 * UCS_KBYTE;
    static pthread_mutex_t mutex  = PTHREAD_MUTEX_INITIALIZER;
    static void *mem_area         = MAP_FAILED;
    static size_t alloc_offset    = 0;
    size_t alloc_size;
    void *result;

    pthread_mutex_lock(&mutex);

    if (mem_area == MAP_FAILED) {
        /* Allocate executable memory block once, and reuse it for
         * subsequent allocations. We assume bistro would not really need
         * more than 'mmap_size' in total, since it's used for limited number
         * of library functions. Also, the memory is never really released, so
         * our allocator is very simple.
         */
        mem_area = mmap(NULL, ucs_align_up_pow2(mmap_size, ucm_get_page_size()),
                        PROT_READ | PROT_WRITE | PROT_EXEC,
                        MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
        if (mem_area == MAP_FAILED) {
            ucm_error("failed to allocated executable memory of %zu bytes: %m",
                      mmap_size);
            result = NULL;
            goto out;
        }
    }

    alloc_size = ucs_align_up_pow2(size, UCS_SYS_PARAGRAPH_SIZE);
    if ((alloc_size + alloc_offset) > mmap_size) {
        result = NULL;
        goto out;
    }

    /* Allocate next memory block in the mmap-ed area */
    result        = UCS_PTR_BYTE_OFFSET(mem_area, alloc_offset);
    alloc_offset += alloc_size;

out:
    pthread_mutex_unlock(&mutex);
    return result;
}

static const char *
ucm_bistro_dump_code(const void *code, size_t length, char *str, size_t max)
{
    const void *code_p = code;
    char *p            = str;
    char *endp         = str + max;

    while (code_p < UCS_PTR_BYTE_OFFSET(code, length)) {
        snprintf(p, endp - p, " %02X",
                 *ucs_serialize_next(&code_p, const uint8_t));
        p += strlen(p);
    }

    return str;
}

ucs_status_t
ucm_bistro_relocate_code(void *dst, const void *src, size_t min_src_length,
                         size_t max_dst_length, size_t *dst_length_p,
                         size_t *src_length_p, const char *symbol,
                         ucm_bistro_relocate_context_t *ctx)
{
    ucs_status_t status;
    char code_buf[64];
    int dladdr_ret;
    Dl_info dli;

    ctx->src_p   = src;
    ctx->dst_p   = dst;
    ctx->dst_end = UCS_PTR_BYTE_OFFSET(dst, max_dst_length);
    ctx->src_end = (void*)UINTPTR_MAX;

    while (ctx->src_p < UCS_PTR_BYTE_OFFSET(src, min_src_length)) {
        status = ucm_bistro_relocate_one(ctx);
        if (status != UCS_OK) {
            goto err;
        }

        if (ctx->src_p > ctx->src_end) {
            status = UCS_ERR_UNSUPPORTED;
            goto err;
        }
    }

    *src_length_p = UCS_PTR_BYTE_DIFF(src, ctx->src_p);
    *dst_length_p = UCS_PTR_BYTE_DIFF(dst, ctx->dst_p);
    return UCS_OK;

err:
    dladdr_ret = dladdr(src, &dli);
    ucm_diag("failed to patch '%s' from %s length %zu code:%s", symbol,
             (dladdr_ret != 0) ? dli.dli_fname : "(unknown)", min_src_length,
             ucm_bistro_dump_code(src, 16, code_buf, sizeof(code_buf)));
    return status;
}

#endif
