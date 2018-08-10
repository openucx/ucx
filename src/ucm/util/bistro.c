/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <sys/mman.h>
#include <dlfcn.h>
#include <string.h>
#include <stdlib.h>

#include <ucm/util/bistro.h>
#include <ucm/util/sys.h>
#include <ucs/sys/math.h>
#include <ucs/arch/cpu.h>
#include <ucs/debug/assert.h>

#define UCM_PROT_READ_WRITE (PROT_READ | PROT_WRITE | PROT_EXEC)
#define UCM_PROT_READ_EXEC  (PROT_READ | PROT_EXEC)

#define UCM_LOOKUP_SYMBOL(_func, _symbol) \
    _func = ucm_bistro_lookup(_symbol);   \
    if (!_func) {                         \
        return UCS_ERR_NO_ELEM;           \
    }

#if defined(__x86_64__) || defined (__aarch64__)
static ucs_status_t ucm_bistro_create_restore_point(void *addr, ucm_bistro_restore_point_h *rp);
#endif

static void *ucm_bistro_page_align(void *ptr)
{
    return (void*)ucs_align_down((uintptr_t)ptr, ucm_get_page_size());
}

static ucs_status_t ucm_bistro_protect(void *addr, size_t len, int prot)
{
    void *aligned = ucm_bistro_page_align(addr);
    size_t size   = addr - aligned + len;

    return mprotect(aligned, size, prot) ? UCS_ERR_INVALID_PARAM : UCS_OK;
}

static void *ucm_bistro_lookup(const char *symbol)
{
    void *addr;

    ucs_assert(symbol != NULL);

    addr = dlsym(RTLD_NEXT, symbol);
    if (!addr) {
        addr = dlsym(RTLD_DEFAULT, symbol);
    }
    return addr;
}

static ucs_status_t ucm_bistro_apply_patch(void *dst, void *patch, size_t len)
{
    ucs_status_t status;

    status = ucm_bistro_protect(dst, len, UCM_PROT_READ_WRITE);
    if (UCS_STATUS_IS_ERR(status)) {
        return status;
    }

    memcpy(dst, patch, len);

    status = ucm_bistro_protect(dst, len, UCM_PROT_READ_EXEC);
    if (!UCS_STATUS_IS_ERR(status)) {
        ucs_memory_cpu_fence();
    }
    return status;
}


/* *******************************************************
 * x86 processors family                                 *
 * ***************************************************** */
#if defined(__x86_64__)

typedef struct ucm_bistro_patch {
    uint8_t mov_r11[2];
    void    *ptr;
    uint8_t jmp_r11[3];
} UCS_S_PACKED ucm_bistro_patch_t;

static const ucm_bistro_patch_t patch_tmpl = {
    .mov_r11 = {0x49, 0xbb},
    .jmp_r11 = {0x41, 0xff, 0xe3}
};

ucs_status_t ucm_bistro_patch(const char *symbol, void *hook,
                              ucm_bistro_restore_point_h *rp)
{
    ucm_bistro_patch_t patch = patch_tmpl;
    ucs_status_t status;
    void *func;

    UCM_LOOKUP_SYMBOL(func, symbol);

    UCS_STATIC_ASSERT(sizeof(patch) == 13);
    UCS_STATIC_ASSERT(ucs_offsetof(ucm_bistro_patch_t, mov_r11) == 0);
    UCS_STATIC_ASSERT(ucs_offsetof(ucm_bistro_patch_t, ptr) == 2);
    UCS_STATIC_ASSERT(ucs_offsetof(ucm_bistro_patch_t, jmp_r11) == 10);

    patch.ptr = hook;

    status = ucm_bistro_create_restore_point(func, rp);
    if (UCS_STATUS_IS_ERR(status)) {
        return status;
    }

    return ucm_bistro_apply_patch(func, &patch, sizeof(patch));
}

/* *******************************************************
 * ARM processors family                                 *
 * ***************************************************** */
#elif defined(__aarch64__)

#define _MOV(_reg, _shift, _val, _opcode) \
    (((_opcode) << 23) + ((uint32_t)(_shift) << 21) + ((uint32_t)((_val) & 0xffff) << 5) + (_reg))

/**
 * @brief Generate a mov immediate instruction
 *
 * @param[in] _reg   register number (0-31)
 * @param[in] _shift shift amount (0-3) * 16-bits
 * @param[in] _value immediate value
 */
#define MOVZ(_reg, _shift, _val) _MOV(_reg, _shift, _val, 0x1a5)

/**
 * @brief Generate a mov immediate with keep instruction
 *
 * @param[in] _reg   register number (0-31)
 * @param[in] _shift shift amount (0-3) * 16-bits
 * @param[in] _value immediate value
 */
#define MOVK(_reg, _shift, _val) _MOV(_reg, _shift, _val, 0x1e5)

#define BR(_reg) ((0xd61f << 16) + ((_reg) << 5))

typedef struct ucm_bistro_patch {
    uint32_t reg3;
    uint32_t reg2;
    uint32_t reg1;
    uint32_t reg0;
    uint32_t br;
} UCS_S_PACKED ucm_bistro_patch_t;

ucs_status_t ucm_bistro_patch(const char *symbol, void *hook,
                              ucm_bistro_restore_point_h *rp)
{
    void *func;
    ucs_status_t status;

    /* r15 is the highest numbered temporary register, assuming this one is safe
     * to use. */
    const uint32_t r15 = 15;
    ucm_bistro_patch_t patch = {
        .reg3 = MOVZ(r15, 3, (uintptr_t)hook >> 48),
        .reg2 = MOVK(r15, 2, (uintptr_t)hook >> 32),
        .reg1 = MOVK(r15, 1, (uintptr_t)hook >> 16),
        .reg0 = MOVK(r15, 0, (uintptr_t)hook),
        .br   = BR(r15)
    };

    UCS_STATIC_ASSERT(sizeof(patch) == 20);
    UCS_STATIC_ASSERT(ucs_offsetof(ucm_bistro_patch_t, reg3) == 0);
    UCS_STATIC_ASSERT(ucs_offsetof(ucm_bistro_patch_t, reg2) == 4);
    UCS_STATIC_ASSERT(ucs_offsetof(ucm_bistro_patch_t, reg1) == 8);
    UCS_STATIC_ASSERT(ucs_offsetof(ucm_bistro_patch_t, reg0) == 12);
    UCS_STATIC_ASSERT(ucs_offsetof(ucm_bistro_patch_t, br) == 16);

    UCM_LOOKUP_SYMBOL(func, symbol);

    status = ucm_bistro_create_restore_point(func, rp);
    if (UCS_STATUS_IS_ERR(status)) {
        return status;
    }

    return ucm_bistro_apply_patch(func, &patch, sizeof(patch));
}

/* *******************************************************
 * POWER-PC processors family                            *
 * ***************************************************** */
#elif defined(__powerpc64__)

// PowerPC instructions used in patching
// Reference: "PowerPC User Instruction Set Architecture"

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
    uint32_t addis;
    uint32_t ori1;
    uint32_t rldicr;
    uint32_t oris;
    uint32_t ori2;
} UCS_S_PACKED ucm_bistro_base_patch_t;

typedef struct ucm_bistro_patch {
    ucm_bistro_base_patch_t super;
    uint32_t mtspr;
    uint32_t bcctr;
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

    UCS_STATIC_ASSERT(ucs_offsetof(ucm_bistro_base_patch_t, addis)  == 0);
    UCS_STATIC_ASSERT(ucs_offsetof(ucm_bistro_base_patch_t, ori1)   == 4);
    UCS_STATIC_ASSERT(ucs_offsetof(ucm_bistro_base_patch_t, rldicr) == 8);
    UCS_STATIC_ASSERT(ucs_offsetof(ucm_bistro_base_patch_t, oris)   == 12);
    UCS_STATIC_ASSERT(ucs_offsetof(ucm_bistro_base_patch_t, ori2)   == 16);

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

    UCS_STATIC_ASSERT(ucs_offsetof(ucm_bistro_patch_t, super) == 0);
    UCS_STATIC_ASSERT(ucs_offsetof(ucm_bistro_patch_t, mtspr) == 20);
    UCS_STATIC_ASSERT(ucs_offsetof(ucm_bistro_patch_t, bcctr) == 24);

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
        if (toc_patch->addis  == nop &&
            toc_patch->ori1   == nop &&
            toc_patch->rldicr == nop &&
            toc_patch->oris   == nop &&
            toc_patch->ori2   == nop) {
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

static void *ucm_bistro_get_text_addr(void *addr) {
#if !defined (_CALL_ELF) || (_CALL_ELF != 2)
    return addr ? *(void**)addr : 0;
#else
    return addr;
#endif
}

ucs_status_t ucm_bistro_patch_toc(const char *symbol, void *hook,
                                  ucm_bistro_restore_point_h *rp,
                                  uint64_t toc)
{
    const uint32_t r11 = 11;
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

    ucm_bistro_fill_patch(&patch, r11, (uintptr_t)hook);

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
        *(*rp) = restore;
    }

    return UCS_OK;
}

ucs_status_t ucm_bistro_restore(ucm_bistro_restore_point_h rp)
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

void *ucm_bistro_restore_addr(ucm_bistro_restore_point_h rp)
{
    ucs_assert(rp != NULL);
    return rp->entry;
}

#else
#  error "Unsupported architecture"
#endif

#if defined(__x86_64__) || defined (__aarch64__)
struct ucm_bistro_restore_point {
    void               *addr; /* address of function to restore */
    ucm_bistro_patch_t patch; /* original function body */
};

static ucs_status_t ucm_bistro_create_restore_point(void *addr, ucm_bistro_restore_point_h *rp)
{
    ucm_bistro_restore_point_t *point;

    if (!rp) {
        return UCS_OK;
    }

    point = malloc(sizeof(*point));
    if (!point) {
        return UCS_ERR_NO_MEMORY;
    }

    point->addr  = addr;
    point->patch = *(ucm_bistro_patch_t*)addr;
    *rp = point;
    return UCS_OK;
}

ucs_status_t ucm_bistro_restore(ucm_bistro_restore_point_h rp)
{
    ucs_status_t status;

    status = ucm_bistro_apply_patch(rp->addr, &rp->patch, sizeof(rp->patch));
    if (!UCS_STATUS_IS_ERR(status)) {
        ucm_bistro_remove_restore_point(rp);
    }
    return status;
}

void *ucm_bistro_restore_addr(ucm_bistro_restore_point_h rp)
{
    ucs_assert(rp != NULL);
    return rp->addr;
}
#endif

ucs_status_t ucm_bistro_remove_restore_point(ucm_bistro_restore_point_h rp)
{
    ucs_assert(rp != NULL);
    free(rp);
    return UCS_OK;
}
