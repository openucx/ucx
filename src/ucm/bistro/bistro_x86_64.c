/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.       ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

/* *******************************************************
 * x86 processors family                                 *
 * ***************************************************** */
#if defined(__x86_64__)

#include <sys/mman.h>
#include <dlfcn.h>
#include <string.h>
#include <stdlib.h>

#include <ucm/bistro/bistro.h>
#include <ucm/bistro/bistro_int.h>
#include <ucm/util/sys.h>
#include <ucs/sys/math.h>
#include <ucs/arch/cpu.h>
#include <ucs/debug/assert.h>


ucs_status_t ucm_bistro_patch(const char *symbol, void *hook,
                              ucm_bistro_restore_point_t **rp)
{
    ucm_bistro_jmp_r11_patch_t patch_jmp_r11   = {
        .mov_r11 = {0x49, 0xbb},
        .jmp_r11 = {0x41, 0xff, 0xe3}
    };
    ucm_bistro_jmp_near_patch_t patch_jmp_near = {
        .jmp_rel = 0xe9
    };
    void *func, *patch, *jmp_base;
    ucs_status_t status;
    ptrdiff_t jmp_disp;
    size_t patch_len;

    UCM_LOOKUP_SYMBOL(func, symbol);

    jmp_base = UCS_PTR_BYTE_OFFSET(func, sizeof(patch_jmp_near));
    jmp_disp = UCS_PTR_BYTE_DIFF(jmp_base, hook);
    if (labs(jmp_disp) < INT32_MAX) {
        /* if 32-bit near jump is possible, use it, since it's a short 5-byte
         * instruction which reduces the chances of racing with other thread
         */
        patch_jmp_near.disp = jmp_disp;
        patch               = &patch_jmp_near;
        patch_len           = sizeof(patch_jmp_near);
    } else {
        patch_jmp_r11.ptr   = hook;
        patch               = &patch_jmp_r11;
        patch_len           = sizeof(patch_jmp_r11);
    }

    status = ucm_bistro_create_restore_point(func, patch_len, rp);
    if (UCS_STATUS_IS_ERR(status)) {
        return status;
    }

    return ucm_bistro_apply_patch(func, patch, patch_len);
}
#endif
