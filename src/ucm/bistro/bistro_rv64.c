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
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>

#include <ucm/bistro/bistro.h>
#include <ucm/bistro/bistro_int.h>
#include <ucm/util/sys.h>
#include <ucs/sys/math.h>
#include <ucs/arch/cpu.h>
#include <ucs/debug/assert.h>

/* Registers numbers to use with the move immediate to register.
  * The destination register is X31 (highest temporary).
  * Register X28-X30 are used for block shifting and masking.
  * Register X0 is always zero
  */
 #define X31 31
 #define X30 30
 #define X0  0

 /**
  * @brief JALR - Add 12 bit immediate to source register, save to destination register, jump and link from destination register
  *
  * @param[in] _reg  register number (0-31), @param[out] _reg register number (0-31), @param[imm] 12 bit immmediate value
  */
 #define JALR(_regs, _regd, _imm) (((_imm) << 20) | ((_regs) << 15) | (0b000 << 12) | ((_regd) << 7) | (0x67))

 /**
  * @brief ADDI - Add 12 bit immediate to source register, save to destination register 
  *
  * @param[in] _reg  register number (0-31), @param[out] _reg register number (0-31), @param[imm] 12 bit immmediate value
  */
 #define ADDI(_regs, _regd, _imm) (((_imm) << 20) | ((_regs) << 15) | (0b000 << 12) | ((_regd) << 7) | (0x13))
 #define ADD(_regs_a, _regs_b, _regd) ((_regs_b << 20) | (_regs_a << 15) | (0b000 << 12) | ((_regd) << 7) | (0x33))

 /**
  * @brief LUI - load upper 20 bit immediate to destination register
  *
  * @param[in] _reg  register number (0-31), @param[out] _reg register number (0-31), @param[imm] 12 bit immmediate value
  */
 #define LUI(_regd, _imm) (((_imm) << 12) | ((_regd) << 7) | (0x37))

 /**
  * @brief SLLI - left-shift immediate number of bits in source register into destination register
  *
  * @param[in] _reg  register number (0-31), @param[out] _reg register number (0-31), @param[imm] 12 bit immmediate value
  */
 #define SLLI(_regs, _regd, _imm) (((_imm) << 20) | ((_regs) << 15) | (0b001 << 12) | ((_regd) << 7) | (0x13))

ucs_status_t ucm_bistro_patch(void *func_ptr, void *hook, const char *symbol,
                              void **orig_func_p,
                              ucm_bistro_restore_point_t **rp)
{
    ucs_status_t status;
    uintptr_t hookp = (uintptr_t)hook;

    ucm_bistro_patch_t patch = {
        .rega = LUI  (X31, ((0xFFFFF << 12) & ( ((hookp) >> 32) + 1 ) ) >> 12),
        .regb = ADDI (X31, X31, ((0xFFF)    & ( ((hookp) >> 32) + 1 ) )      ),
        .regc = LUI  (X30, ((0xFFFFF << 12) & ( (((hookp)) + 1)     ) ) >> 12),
        .regd = SLLI (X31, X31, 32),
        .rege = ADD  (X30, X31, X31),
        .regf = JALR (X31, X0, ((0xFFF)     & ( ((hookp)) + 1)))
    };

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
