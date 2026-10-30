/**
 * Copyright (C) Loongson Technology Corporation Limited, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#if defined(__loongarch_lp64)

#include <ucm/bistro/bistro.h>
#include <ucm/bistro/bistro_int.h>

#include <stdint.h>

/* LoongArch general-purpose register numbers used by the trampoline */
#define UCM_BISTRO_LA_REG_ZERO 0
#define UCM_BISTRO_LA_REG_T0   12
#define UCM_BISTRO_LA_REG_T2   14

/**
 * JIRL - Add a 16-bit immediate to a source register, save the return address
 * into a destination register, and jump to the source register.
 *
 * Encoding: opcode | offs16 | rj | rd
 *
 * @param[in] _regd destination register number (0-31)
 * @param[in] _regj source register number (0-31)
 * @param[in] _imm  16-bit branch offset, word-scaled (in units of 4-byte
 *                  instructions; the hardware computes rj + (offs << 2))
 */
#define UCM_BISTRO_LA_JIRL(_regd, _regj, _imm) \
    (0x4c000000 | ((_imm) << 10) | ((_regj) << 5) | (_regd))

/**
 * B - unconditional branch (jump to self when the immediate is 0)
 *
 * Encoding: opcode | offs[15:0] | offs[25:16]
 *
 * @param[in] _imm 26-bit branch offset, word-scaled (in units of 4-byte
 *                 instructions; the hardware computes pc + (offs << 2))
 */
#define UCM_BISTRO_LA_B(_imm) \
    (0x50000000 | (((_imm) & 0xffff) << 10) | (((_imm) >> 16) & 0x3ff))

/**
 * PCADDU12I - Add an upper 20-bit immediate (shifted by 12) to the PC
 *
 * Encoding: opcode | si20 | rd
 *
 * @param[in] _regd destination register number (0-31)
 * @param[in] _imm  20-bit immediate value
 */
#define UCM_BISTRO_LA_PCADDU12I(_regd, _imm) \
    (0x1c000000 | ((_imm) << 5) | (_regd))

/**
 * LD - Load a doubleword from memory using a register plus a 12-bit immediate
 *
 * Encoding: opcode | si12 | rj | rd
 *
 * @param[in] _regd destination register number (0-31)
 * @param[in] _regj source register number (0-31)
 * @param[in] _imm  12-bit immediate value
 */
#define UCM_BISTRO_LA_LD(_regd, _regj, _imm) \
    (0x28c00000 | ((_imm) << 10) | ((_regj) << 5) | (_regd))

/**
 * BREAK - trap with an immediate code, used to stop control flow
 *
 * @param[in] _code 15-bit code
 */
#define UCM_BISTRO_LA_BREAK(_code) \
    (0x002a0000 | ((_code) & 0x7fff))

void ucm_bistro_patch_lock(void *dst)
{
    static const ucm_bistro_lock_t self_jmp = {
        .j = UCM_BISTRO_LA_B(0)
    };

    ucm_bistro_modify_code(dst, &self_jmp);
}

ucs_status_t ucm_bistro_patch(void *func_ptr, void *hook, const char *symbol,
                              void **orig_func_p,
                              ucm_bistro_restore_point_t **rp)
{
    ucs_status_t status;

    /*
     * The trampoline below computes the address of the patched instruction
     * (pcaddu12i), loads the hook address stored right after the four
     * instruction words (at offset 0x10), and jumps to it:
     *
     *   pcaddu12i $t0, 0          # $t0 = address of this instruction
     *   ld.d      $t2, $t0, 0x10  # $t2 = *(patch + 0x10) = hook
     *   jirl      $zero, $t2, 0   # jump to hook
     */
    ucm_bistro_patch_t patch = {
        .pcaddu12i = UCM_BISTRO_LA_PCADDU12I(UCM_BISTRO_LA_REG_T0, 0),
        .ld        = UCM_BISTRO_LA_LD(UCM_BISTRO_LA_REG_T2,
                                       UCM_BISTRO_LA_REG_T0, 0x10),
        .jirl      = UCM_BISTRO_LA_JIRL(UCM_BISTRO_LA_REG_ZERO,
                                        UCM_BISTRO_LA_REG_T2, 0),
        .spare     = UCM_BISTRO_LA_BREAK(0),
        .address   = (uintptr_t)hook
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
