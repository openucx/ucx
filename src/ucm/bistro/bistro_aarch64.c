/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.       ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

/* *******************************************************
 * ARM processors family                                 *
 * ***************************************************** */
#if defined(__aarch64__)

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


/* Register number used to store indirect jump address.
 * r15 is the highest numbered temporary register, assuming this one is safe
 * to use. */
#define R15 15

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

/**
 * @brief Branch to address stored in register
 *
 * @param[in] _reg   register number (0-31)
 */
#define BR(_reg) ((0xd61f << 16) + ((_reg) << 5))

ucs_status_t ucm_bistro_patch(const char *symbol, void *hook,
                              ucm_bistro_restore_point_t **rp)
{
    void *func;
    ucs_status_t status;

    ucm_bistro_patch_t patch = {
        .reg3 = MOVZ(R15, 3, (uintptr_t)hook >> 48),
        .reg2 = MOVK(R15, 2, (uintptr_t)hook >> 32),
        .reg1 = MOVK(R15, 1, (uintptr_t)hook >> 16),
        .reg0 = MOVK(R15, 0, (uintptr_t)hook),
        .br   = BR(R15)
    };

    UCM_LOOKUP_SYMBOL(func, symbol);

    status = ucm_bistro_create_restore_point(func, rp);
    if (UCS_STATUS_IS_ERR(status)) {
        return status;
    }

    return ucm_bistro_apply_patch(func, &patch, sizeof(patch));
}

#endif
