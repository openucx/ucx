/**
 * Copyright (C) Tactical Computing Labs, LLC. 2022. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

/* *******************************************************
 * RISC-V processors family                              *
 * ***************************************************** */
#if defined(__riscv)

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


/* Registers numbers to use with the move immediate to register.
 * The destination register is X31 (highest temporary).
 * Register X28-X30 are used for block shifting and masking.
 * Register X0 is always zero */
#define X28 28
#define X29 29
#define X30 30
#define X31 31
#define X0  0

/**
 * @brief Performs an OR immediate operation of the reigster _regRS1 and the
 *        immediate value _val storing the result to _regRD
 *
 * @param[in] _regRD  register number (0-31)
 * @param[in] _regRS1 register number (0-31)
 * @param[in] _val    immediate value
 */
#define ORI(_regRD, _regRS1, _val) (((_val) << 20) + ((_regRS1) << 15) + \
                                    (0x6 << 12) + ((_regRD) << 7) + (0x13))

/**
 * @brief Peforms a logical OR operation of the registers _regRS1 and _regRS2
 *        and stores the result to _regRD
 *
 * @param[in] _regRD  register number (0-31)
 * @param[in] _regRS1 register number (0-31)
 * @param[in] _regRS2 register number (0-31)
 */
#define OR(_regRD, _regRS1, _regRS2) (((_regRS2) << 20) + ((_regRS1) << 15) + \
                                      (0x6 << 12) + ((_regRD) << 7)+ (0x33))

/**
 * @brief Performs a shift left logical immediate of the value in register
 *        _regRS1 by the immedate value _val and stores to _regRD
 *
 * @param[in] _regRD  register number (0-31)
 * @param[in] _regRS1 register number (0-31)
 * @param[in] _val    immediate value
 */
#define SLLI(_regRD, _regRS1, _val) (((_val) << 20) + ((_regRS1) << 15) + \
                                     (1<<12) + ((_regRD) << 7) + (0x13))

/**
 * @brief Performs an add immediate of the value and stores to the
 *        target register
 *
 * @param[in] _reg  register number (0-31)
 * @param[in] _val  immediate value
 */
#define ADDI(_reg, _val) (((_val) << 20)((_reg) << 7)+ (0x13))

/**
 * @brief Branch to address stored in register
 *
 * @param[in] _reg  register number (0-31)
 */
#define BR(_reg) (((_reg) << 15) + (0x67))

ucs_status_t ucm_bistro_patch(void *func_ptr, void *hook, const char *symbol,
                              void **orig_func_p,
                              ucm_bistro_restore_point_t **rp)
{
    ucm_bistro_patch_t patch = {
        .reg14  = ADDI(X31, X0, (uintptr_t)(hook>>52),
        .reg13  = SLLI(X31, X31, 52),
        .reg12  = ADDI(X30, X0, ((uintptr_t)(hook>>40)&0b111111111111)),
        .reg11  = SLLI(X30, X30, 40),
        .reg10  = OR(X31, X31, X30),
        .reg9   = ADDI(X29, X0, ((uintptr_t)(hook>>28)&0b111111111111))
        .reg8   = SLLI(X29, X29, 28),
        .reg7   = OR(X31, X31, X29),
        .reg6   = ADDI(X28, X0, ((uintptr_t)(hook>>16)&0b111111111111)),
        .reg5   = SLLI(X28, X28, 16),
        .reg4   = OR(X31, X31, X28),
        .reg3   = ADDI(X30, X0, ((uintptr_t)(hook>>4)&0b111111111111)),
        .reg2   = SLLI(X30, X30, 4),
        .reg1   = OR(X31, X31, X30),
        .reg0   = ORI(X31, X31, (uintptr_t)hook&0b1111),
        .br     = BR(X31)
    };
    ucs_status_t status;
    if (orig_func_p != NULL) {
        return UCS_ERR_UNSUPPORTED;
    }

    status = ucm_bistro_create_restore_point(func_ptr, sizeof(patch), rp);
    if (UCS_STATUS_IS_ERR(status)) {
        return status;
    }

    return ucm_bistro_apply_patch(func_ptr, &patch, sizeof(patch));
}
#endif
