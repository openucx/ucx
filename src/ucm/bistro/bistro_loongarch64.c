/**
 * Copyright (C) Xing Li, Dandan Zhang, 2024. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#if defined(__loongarch64)

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

#define T0 12
#define T2 14
#define RA  1
#define ZERO  0

/**
  * @brief JIRL - Add 16 bit immediate to source register, save to destination
  * register, jump and link from destination register
  *
  * @param[in] _regd source register number (0-31)
  * @param[in] _regj destination register number (0-31)
  * @param[in] _imm 16 bit immmediate value
  */
#define JIRL(_regd, _regj, _imm) \
    (((0x13) << 26 ) | ((_imm) << 10) | ((_regj) << 5) | (_regd))
/**
  * @brief B - Indirect jump
  *
  * @param[in] _imm 26 bit immmediate value
  */
#define B(_imm) \
	((0x14) << 26) | (((_imm) & 0xffff) << 10) | ((_imm) >>16)

/**
  * @brief PCADDU12I - Add upper intermediate to PC
  *
  * @param[in] _regd register number (0-31)
  * @param[in] _imm 20 bit immmediate value
  */
#define PCADDU12I(_regd, _imm) (((0xe) << 25) | ((_imm) << 5) | (_regd))

/**
  * @brief LD - Load from memory with address from register plus immediate
  *
  * @param[in] _regs source register number (0-31)
  * @param[in] _regd destination register number (0-31)
  * @param[in] _imm 12 bit immmediate value
  */
#define LD(_regd, _regj, _imm) \
    (((0xa3) << 22) | ((_imm) << 10) | ((_regj) << 5) | (_regd))

void ucm_bistro_patch_lock(void *dst)
{
    static const ucm_bistro_lock_t self_jmp = {
        .j = B(0)
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
        .pcaddu12i = PCADDU12I(T0, 0),
        .ld        = LD(T2, T0, 0x10),
        .jirl      = JIRL(0, T2, 0),
        .spare     = 0,
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
