/**
 * Copyright (C) Tactical Computing Labs, LLC. 2022. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#if defined(__riscv)

#include <ucs/arch/cpu.h>
#include <ucm/bistro/bistro.h>
#include <ucm/bistro/bistro_int.h>
#include <ucs/debug/assert.h>
#include <ucs/sys/math.h>
#include <ucm/util/sys.h>

#include <assert.h>
#include <dlfcn.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#define X31 31
#define X0  0

/**
  * @brief JALR - Add 12 bit immediate to source register, save to destination
  * register, jump and link from destination register
  *
  * @param[in] _regs source register number (0-31)
  * @param[in] _regd destination register number (0-31)
  * @param[in] _imm 12 bit immediate value
  */
#define JALR(_regs, _regd, _imm) \
    (((_imm) << 20) | ((_regs) << 15) | (0b000 << 12) | ((_regd) << 7) | (0x67))

/**
  * @brief C_J - Indirect jump (using compressed instruction)
  *
  * @param[in] _imm 12 bit immediate value
  */
#define C_J(_imm) \
    ((0b101) << 13 | ((_imm >> 1) << 2) | (0b01))

/**
  * @brief AUIPIC - Add upper intermediate to PC
  *
  * @param[in] _regd register number (0-31)
  * @param[in] _imm 12 bit immediate value
  */
#define AUIPC(_regd, _imm) (((_imm) << 12) | ((_regd) << 7) | (0x17))

/**
  * @brief LD - Load from memory with address from register plus immediate
  *
  * @param[in] _regs source register number (0-31)
  * @param[in] _regd destination register number (0-31)
  * @param[in] _imm 12 bit immediate value
  */
#define LD(_regs, _regd, _imm) \
    (((_imm) << 20) | ((_regs) << 15) | (0b011 << 12) | ((_regd) << 7) | (0x3))

void ucm_bistro_patch_lock(void *dst)
{
    static const ucm_bistro_lock_t self_jmp = {
        .j = C_J(0)
    };
    ucm_bistro_modify_code(dst, &self_jmp);
}

ucs_status_t ucm_bistro_patch(void *func_ptr, void *hook, const char *symbol,
                              void **orig_func_p,
                              ucm_bistro_restore_point_t **rp)
{
    ucs_status_t status;
    ucm_bistro_patch_t patch;

    patch = (ucm_bistro_patch_t) {
        .auipc   = AUIPC(X31, 0),
        .ld      = LD(31, 31, 0x10),
        .jalr    = JALR(X31, X0, 0),
        .spare   = 0,
        .address = (uintptr_t)hook
    };

    if (orig_func_p != NULL) {
        return UCS_ERR_UNSUPPORTED;
    }

    status = ucm_bistro_create_restore_point(func_ptr, sizeof(patch), rp);
    if (UCS_STATUS_IS_ERR(status)) {
        return status;
    }

    return ucm_bistro_apply_patch_atomic(func_ptr, &patch, sizeof(patch));
}

ucs_status_t ucm_bistro_relocate_one(ucm_bistro_relocate_context_t *ctx)
{
    return UCS_ERR_UNSUPPORTED;
}

#endif
