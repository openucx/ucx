/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.       ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

/* *******************************************************
 * POWER-PC processors family                            *
 * ***************************************************** */
#if defined (__powerpc64__)

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

/* PowerPC instructions used in patching                  */
/* Reference: "PowerPC User Instruction Set Architecture" */

/* Use r11 register for jump address */
#define R11 11

#define OPCODE(_rt, _rs, _op) \
     (((_op) << 26) + ((_rt) << 21) + ((_rs) << 16))

#define OP0(_rt, _rs, _ui, _op) \
     (OPCODE(_rt, _rs, _op) + ((_ui) & 0xffff))

#define MTSPR(_spr, _rs) \
    (OPCODE(_rs, (_spr) & 0x1f, 31) + (((_spr) & ~UCS_MASK(5)) << 6) + (467 << 1))

#define BCCTR(_bo, _bi, _bh) \
    (OPCODE(_bo, _bi, 19) + ((_bh) << 11) + (528<<1))

#define RLDICR(_rt, _rs, _sh, _mb) \
    (OPCODE(_rs, _rt, 30) + (((_sh) & UCS_MASK(5)) << 11) + ((_sh & ~UCS_MASK(5)) >> 4) + \
    (((_mb) & UCS_MASK(5)) << 6) + ((_mb) && ~UCS_MASK(5)) + UCS_BIT(2))

#define ADDIS(_rt, _rs, _ui) OP0(_rt, _rs, _ui, 15)
#define ORI(_rt, _rs, _ui) OP0(_rs, _rt, _ui, 24)
#define ORIS(_rt, _rs, _ui) OP0(_rs, _rt, _ui, 25)

typedef struct ucm_bistro_base_patch {
    uint32_t addis;    /* lis     r11,(addr >> 48)     */
    uint32_t ori1;     /* ori     r11,r11,(addr >> 32) */
    uint32_t rldicr;   /* rldicr  r11,r11,32,31        */
    uint32_t oris;     /* oris    r11,r11,(addr >> 16) */
    uint32_t ori2;     /* ori     r11,r11,addr         */
} UCS_S_PACKED ucm_bistro_base_patch_t;

typedef struct ucm_bistro_patch {
    ucm_bistro_base_patch_t super;
    uint32_t                mtspr;    /* mtspr r11 */
    uint32_t                bcctr;    /* bcctr     */
} UCS_S_PACKED ucm_bistro_patch_t;

struct ucm_bistro_restore_point {
    void                    *entry;
    void                    *hook;
    ucm_bistro_base_patch_t hook_patch;
    void                    *func;
    ucm_bistro_patch_t      func_patch;
};

static void ucm_bistro_fill_base_patch(ucm_bistro_base_patch_t *patch,
                                       uint32_t reg, uintptr_t value)
{
    ucs_assert(patch != NULL);

    patch->addis  = ADDIS ( reg, 0,   (value >> 48));
    patch->ori1   = ORI   ( reg, reg, (value >> 32));
    patch->rldicr = RLDICR( reg, reg, 32, 31);
    patch->oris   = ORIS  ( reg, reg, (value >> 16));
    patch->ori2   = ORI   ( reg, reg, (value >>  0));
}

static void ucm_bistro_fill_patch(ucm_bistro_patch_t *patch,
                                  uint32_t reg, uintptr_t value)
{
    ucs_assert(patch != NULL);

    ucm_bistro_fill_base_patch(&patch->super, reg, value);

    patch->mtspr = MTSPR(9, reg);   /* 9 = CTR     */
    patch->bcctr = BCCTR(20, 0, 0); /* 20 = always */
}

static ucs_status_t ucm_bistro_patch_hook(void *hook, ucm_bistro_restore_point_t *rp,
                                          uint64_t toc)
{
    const uint32_t nop = 0x60000000;
    uint32_t *toc_ptr;
    ucm_bistro_base_patch_t *toc_patch;
    ucm_bistro_base_patch_t patch;

    /* locate reserved code space in hook function */
    for (toc_ptr = hook;; toc_ptr++) {
        toc_patch = (ucm_bistro_base_patch_t*)toc_ptr;
        if ((toc_patch->addis  == nop) &&
            (toc_patch->ori1   == nop) &&
            (toc_patch->rldicr == nop) &&
            (toc_patch->oris   == nop) &&
            (toc_patch->ori2   == nop)) {
            break;
        }
    }

    if (rp) {
        rp->hook       = toc_ptr;
        rp->hook_patch = *toc_patch;
    }

    ucm_bistro_fill_base_patch(&patch, 2, toc);
    return ucm_bistro_apply_patch(toc_ptr, &patch, sizeof(patch));
}

static void *ucm_bistro_get_text_addr(void *addr)
{
#if !defined (_CALL_ELF) || (_CALL_ELF != 2)
    return addr ? *(void**)addr : 0;
#else
    return addr;
#endif
}

ucs_status_t ucm_bistro_patch_toc(const char *symbol, void *hook,
                                  ucm_bistro_restore_point_t **rp,
                                  uint64_t toc)
{
    ucs_status_t status;
    void *func;
    ucm_bistro_restore_point_t restore;
    ucm_bistro_patch_t patch;

    UCM_LOOKUP_SYMBOL(func, symbol);

    restore.entry = func;

    func = ucm_bistro_get_text_addr(func);
    hook = ucm_bistro_get_text_addr(hook);

    status = ucm_bistro_patch_hook(hook, &restore, toc);
    if (UCS_STATUS_IS_ERR(status)) {
        return status;
    }

#if defined(_CALL_ELF) && (_CALL_ELF == 2)
    func += 8;
    hook += 8;
#endif

    ucm_bistro_fill_patch(&patch, R11, (uintptr_t)hook);

    restore.func       = func;
    restore.func_patch = *(ucm_bistro_patch_t*)func;

    status = ucm_bistro_apply_patch(func, &patch, sizeof(patch));
    if (UCS_STATUS_IS_ERR(status)) {
        return status;
    }

    if (rp) {
        *rp = malloc(sizeof(restore));
        if (!(*rp)) {
            return UCS_ERR_NO_MEMORY;
        }
        **rp = restore;
    }

    return UCS_OK;
}

ucs_status_t ucm_bistro_restore(ucm_bistro_restore_point_t *rp)
{
    ucs_status_t status;

    ucs_assert(rp != NULL);

    status = ucm_bistro_apply_patch(rp->func, &rp->func_patch, sizeof(rp->func_patch));
    if (UCS_STATUS_IS_ERR(status)) {
        return status;
    }

    status = ucm_bistro_apply_patch(rp->hook, &rp->hook_patch, sizeof(rp->hook_patch));
    if (!UCS_STATUS_IS_ERR(status)) {
        ucm_bistro_remove_restore_point(rp);
    }

    return status;
}

void *ucm_bistro_restore_addr(ucm_bistro_restore_point_t *rp)
{
    ucs_assert(rp != NULL);
    return rp->entry;
}

#endif
