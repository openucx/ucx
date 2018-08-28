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

static const ucm_bistro_patch_t patch_tmpl = {
    .mov_r11 = {0x49, 0xbb},
    .jmp_r11 = {0x41, 0xff, 0xe3}
};

ucs_status_t ucm_bistro_patch(const char *symbol, void *hook,
                              ucm_bistro_restore_point_t **rp)
{
    ucm_bistro_patch_t patch = patch_tmpl;
    ucs_status_t status;
    void *func;

    UCM_LOOKUP_SYMBOL(func, symbol);

    patch.ptr = hook;

    status = ucm_bistro_create_restore_point(func, rp);
    if (UCS_STATUS_IS_ERR(status)) {
        return status;
    }

    return ucm_bistro_apply_patch(func, &patch, sizeof(patch));
}
#endif
