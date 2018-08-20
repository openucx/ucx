/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.       ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCM_BISTRO_BISTRO_INT_H_
#define UCM_BISTRO_BISTRO_INT_H_

#include <sys/mman.h>
#include <dlfcn.h>
#include <string.h>
#include <stdlib.h>

#include <ucm/bistro/bistro.h>
#include <ucm/util/sys.h>
#include <ucm/util/log.h>
#include <ucs/sys/math.h>
#include <ucs/arch/cpu.h>
#include <ucs/debug/assert.h>

#define UCM_PROT_READ_WRITE_EXEC (PROT_READ | PROT_WRITE | PROT_EXEC)
#define UCM_PROT_READ_EXEC       (PROT_READ | PROT_EXEC)

#define UCM_LOOKUP_SYMBOL(_func, _symbol) \
    _func = ucm_bistro_lookup(_symbol);   \
    if (!_func) {                         \
        return UCS_ERR_NO_ELEM;           \
    }

#if defined(__x86_64__)
typedef struct ucm_bistro_patch {
    uint8_t mov_r11[2];  /* mov %r11, addr */
    void    *ptr;
    uint8_t jmp_r11[3];  /* jmp r11        */
} UCS_S_PACKED ucm_bistro_patch_t;
#elif defined (__aarch64__)
typedef struct ucm_bistro_patch {
    uint32_t reg3;  /* movz    x15, addr, lsl #48 */
    uint32_t reg2;  /* movk    x15, addr, lsl #32 */
    uint32_t reg1;  /* movk    x15, addr, lsl #16 */
    uint32_t reg0;  /* movk    x15, addr          */
    uint32_t br;    /* br      x15                */
} UCS_S_PACKED ucm_bistro_patch_t;
#endif

ucs_status_t ucm_bistro_apply_patch(void *dst, void *patch, size_t len);

#if defined(__x86_64__) || defined (__aarch64__)
ucs_status_t ucm_bistro_create_restore_point(void *addr, ucm_bistro_restore_point_h *rp);
#endif

static inline
void *ucm_bistro_lookup(const char *symbol)
{
    void *addr;

    ucs_assert(symbol != NULL);

    addr = dlsym(RTLD_NEXT, symbol);
    if (!addr) {
        addr = dlsym(RTLD_DEFAULT, symbol);
    }
    return addr;
}
#endif
