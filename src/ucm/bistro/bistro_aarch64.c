/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2018. ALL RIGHTS RESERVED.
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
#include <ucs/type/serialize.h>


/* Caller-saved registers are x9-x15 */
#define UCM_AARCH64_TMP_REGS \
    (UCS_BIT(9) | UCS_BIT(10) | UCS_BIT(11) | UCS_BIT(12) | UCS_BIT(13) | \
     UCS_BIT(14) | UCS_BIT(15))


/**
 * @brief Generate a generic mov instruction
 *
 * @param[in] _reg     register number (0-31)
 * @param[in] _shift   shift amount (0-3) * 16-bits
 * @param[in] _value   immediate value
 * @param[in] _opcode  move instruction opcode
 */
#define UCM_AARCH64_MOV(_reg, _shift, _val, _opcode) \
    ((((ucm_bistro_inst_t)_opcode) << 23) | \
     ((ucm_bistro_inst_t)(_shift) << 21) | \
     ((ucm_bistro_inst_t)((_val)&0xffff) << 5) | (_reg))

/**
 * @brief Generate a mov immediate instruction
 *
 * @param[in] _reg   register number (0-31)
 * @param[in] _shift shift amount (0-3) * 16-bits
 * @param[in] _value immediate value
 */
#define UCM_AARCH64_MOVZ(_reg, _shift, _val) \
    UCM_AARCH64_MOV(_reg, _shift, _val, 0x1a5)

/**
 * @brief Generate a mov immediate with keep instruction
 *
 * @param[in] _reg   register number (0-31)
 * @param[in] _shift shift amount (0-3) * 16-bits
 * @param[in] _value immediate value
 */
#define UCM_AARCH64_MOVK(_reg, _shift, _val) \
    UCM_AARCH64_MOV(_reg, _shift, _val, 0x1e5)

/**
 * @brief Generate a load-to-register instruction
 *
 * @param [in] _reg    register number to load the value to (0-31)
 * @param [in] _label  memory label to load the value from
 *
 * | 01 | 011 | 0 | 00 | imm19 | Rt |
 *   opc
 */
#define UCM_AARCH64_LDR_LITERAL(_reg, _label) \
    (((ucm_bistro_inst_t)0x58 << 24) | ((_label) << 5) | (_reg))

/**
 * @brief Generate a branch-to-register instruction
 *
 * @param [in] _reg   register number to branch to (0-31)
 *
 * | 1101011 | 0 | 0 | 00 | 11111 | 0000 | 0 | 0 | Rn | 00000 |
 *             Z       op                  A   M        Rm
 */
#define UCM_AARCH64_BR(_reg) \
    (((ucm_bistro_inst_t)0xd61f << 16) + ((_reg) << 5))

#define UCM_AARCH64_B(_off) (((ucm_bistro_inst_t)0x14 << 24) | (_off))

/* Indirect jump code sequence */
typedef struct {
    ucm_bistro_inst_t ldr;  /* ldr <reg>, label */
    ucm_bistro_inst_t br;   /* br  <reg> */
    uint64_t          dest; /* jump destination */
} UCS_S_PACKED ucm_bistro_jmp_indirect_t;

/* 64-bit load immediate code sequence */
typedef struct ucm_bistro_load64 {
    ucm_bistro_inst_t reg3; /* movz    reg, addr, lsl #48 */
    ucm_bistro_inst_t reg2; /* movk    reg, addr, lsl #32 */
    ucm_bistro_inst_t reg1; /* movk    reg, addr, lsl #16 */
    ucm_bistro_inst_t reg0; /* movk    reg, addr          */
} UCS_S_PACKED ucm_bistro_load64_t;

/* Context for code relocation, that remembers which registers were used */
typedef struct {
    ucm_bistro_relocate_context_t super;
    uint32_t                      avail_regs;
} ucm_bistro_aarch64_relocate_context_t;


static ptrdiff_t ucm_bistro_sign_extend(uint64_t value, unsigned bits)
{
    unsigned shift = 64 - bits;

    /* Shift right to push the sign bit to MSB position, then shift-left a
       signed type to force sign extension by arithmetic shift */
    return ((ptrdiff_t)value << shift) >> shift;
}

static void ucm_bistro_init_jmp_indirect(ucm_bistro_jmp_indirect_t *jmp,
                                         void *address, uint8_t regno)
{
    ptrdiff_t offset;

    /* Calculate the label as the difference between the address of the
       ldr instruction and the address of the pointer to the jump target
       divided by 4 (since the label is extended by two zero bits) */
    offset = ucs_offsetof(ucm_bistro_jmp_indirect_t, dest) -
             ucs_offsetof(ucm_bistro_jmp_indirect_t, ldr);
    ucm_assertv_always(offset > 0, "offset=%ld", offset);

    jmp->ldr  = UCM_AARCH64_LDR_LITERAL(regno, offset / 4);
    jmp->br   = UCM_AARCH64_BR(regno);
    jmp->dest = (uintptr_t)address;
}

static void ucm_bistro_init_load64(ucm_bistro_load64_t *load64, uint8_t regno,
                                   uint64_t value)
{
    load64->reg3 = UCM_AARCH64_MOVZ(regno, 3, value >> 48);
    load64->reg2 = UCM_AARCH64_MOVK(regno, 2, value >> 32);
    load64->reg1 = UCM_AARCH64_MOVK(regno, 1, value >> 16);
    load64->reg0 = UCM_AARCH64_MOVK(regno, 0, value);
}

ucs_status_t ucm_bistro_relocate_one(ucm_bistro_relocate_context_t *ctx)
{
    ucm_bistro_aarch64_relocate_context_t *aarch64_ctx =
            ucs_derived_of(ctx, ucm_bistro_aarch64_relocate_context_t);
    uint64_t immlo, immhi, imm;
    ucm_bistro_load64_t load64;
    ucm_bistro_inst_t inst;
    const void *copy_src;
    size_t dst_length;
    uint64_t inst_pc;
    uint8_t regno;

    /* Read next opcode */
    inst_pc = (uintptr_t)aarch64_ctx->super.src_p;
    inst    = *ucs_serialize_next(&aarch64_ctx->super.src_p, const uint32_t);

    if (/* STP post-index     | x 0 1 0 1 0 0 0 1 0 imm7 Rt2 Rn Rt */
        ((inst & 0xffc00000) == 0xa8800000) ||
        /* STP pre-index      | x 0 1 0 1 0 0 1 1 0 imm7 Rt2 Rn Rt */
        ((inst & 0xffc00000) == 0xa9800000) ||
        /* STP signed offset  | x 0 1 0 1 0 0 1 0 0 imm7 Rt2 Rn Rt */
        ((inst & 0xffc00000) == 0xa9000000)) {
        copy_src                 = &inst;
        dst_length               = sizeof(inst);
        regno                    = (inst >> 5) & UCS_MASK(5);
        aarch64_ctx->avail_regs &= ~UCS_BIT(regno);
    } else if (
        /* CMP shft register  | sf 1 1 0 1 0 1 1 shf 0 Rm imm6        Rn 1 1 1 1 1
           CMP ext register   | sf 1 1 0 1 0 1 1 0 0 1 Rm option imm3 Rn 1 1 1 1 1 */
        (inst & 0x7fc0001f) == 0x6b00001f) {
        copy_src   = &inst;
        dst_length = sizeof(inst);
    } else if (
        /* MOVK               | sf 0 0 1 0 0 1 0 1 hw imm16 Rd */
        ((inst & 0x7f800000) == 0x72800000) ||
        /* MOV register       | sf 0 1 0 1 0 1 0 0 0 0 Rm 0 0 0 0 0 0 1 1 1 1 1 Rd */
        ((inst & 0x7fe0ffe0) == 0x2a0003e0) ||
        /* MOV to/from SP     | sf 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 Rn Rd */
        ((inst & 0x7ffffc00) == 0x11000000) ||
        /* MOV wide immediate | sf 1 0 1 0 0 1 0 1 hw imm16 Rd */
        ((inst & 0x7f800000) == 0x52800000) ||
        /* LDR imm post-index | 1 x 1 1 1 0 0 0 0 1 0 imm9 0 1 Rn Rt
           LDR imm pre-index  | 1 x 1 1 1 0 0 0 0 1 0 imm9 1 1 Rn Rt */
        ((inst & 0xbfe00400) == 0xb8400400) ||
        /* LDR unsigned offset | 1 x 1 1 1 0 0 1 0 1 imm12 Rn Rt */
        ((inst & 0xbfc00000) == 0xb9400000)) {
        copy_src                 = &inst;
        dst_length               = sizeof(inst);
        regno                    = inst & UCS_MASK(5);
        aarch64_ctx->avail_regs &= ~UCS_BIT(regno); /* Rt/Rd may be modified */
    } else if (/* ADRP | 1 immlo 1 0 0 0 0 immhi Rd */
               (inst & 0x9f000000) == 0x90000000) {
        /* Translate ADRP to a 64-bit load immediate of the actual address */
        immlo = (inst >> 29) & UCS_MASK(2);
        immhi = (inst >> 5) & UCS_MASK(19);
        regno = inst & UCS_MASK(5);
        imm   = (immhi << 2) | immlo;
        ucm_bistro_init_load64(&load64, regno,
                               inst_pc + ucm_bistro_sign_extend(imm << 12, 33));

        copy_src                 = &load64;
        dst_length               = sizeof(load64);
        aarch64_ctx->avail_regs &= ~UCS_BIT(regno);
    } else {
        return UCS_ERR_UNSUPPORTED;
    }

    if (UCS_PTR_BYTE_OFFSET(aarch64_ctx->super.dst_p, dst_length) >
        aarch64_ctx->super.dst_end) {
        return UCS_ERR_BUFFER_TOO_SMALL;
    }

    /* Copy 'dst_length' bytes to aarch64_ctx->dst_p and advance it */
    memcpy(ucs_serialize_next_raw(&aarch64_ctx->super.dst_p, void, dst_length),
           copy_src, dst_length);
    return UCS_OK;
}

static ucs_status_t
ucm_bistro_relocate_func(const void *func_ptr, size_t patch_len,
                         const char *symbol, void **orig_func_p)
{
    ucm_bistro_aarch64_relocate_context_t aarch64_ctx;
    size_t code_len, prefix_len, max_code_len;
    ucs_status_t status;
    void *orig_func;
    uint8_t regno;

    /* Allocate executable page, calculate the patch size according to worse
       case scenario when need to translate ADRP */
    max_code_len = (patch_len *
                    (sizeof(ucm_bistro_load64_t) / sizeof(ucm_bistro_inst_t))) +
                   sizeof(ucm_bistro_jmp_indirect_t);
    orig_func    = ucm_bistro_allocate_code(max_code_len +
                                            sizeof(ucm_bistro_jmp_indirect_t));
    if (orig_func == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    /* Copy and translate code from 'func_ptr' to 'orig_func->code'.
       'code_len' is the code size at destination buffer, and 'prefix_len' is
       how many bytes were translated from 'func_ptr'. */
    aarch64_ctx.avail_regs = UCM_AARCH64_TMP_REGS;
    status = ucm_bistro_relocate_code(orig_func, func_ptr, patch_len,
                                      max_code_len, &code_len, &prefix_len,
                                      symbol, &aarch64_ctx.super);
    if (status != UCS_OK) {
        return UCS_ERR_UNSUPPORTED;
    }

    if (aarch64_ctx.avail_regs == 0) {
        /* Cannot find register for saving jump address */
        return UCS_ERR_UNSUPPORTED;
    }

    ucm_debug("'%s' at %p code length %zu/%zu prefix length %zu regs 0x%x",
              symbol, func_ptr, code_len, patch_len, prefix_len,
              aarch64_ctx.avail_regs);

    /* Indirect jump from replacement code to original function */
    ucm_assert((code_len + sizeof(ucm_bistro_jmp_indirect_t)) <= max_code_len);
    regno = ucs_ffs32(aarch64_ctx.avail_regs);
    ucm_bistro_init_jmp_indirect(UCS_PTR_BYTE_OFFSET(orig_func, code_len),
                                 UCS_PTR_BYTE_OFFSET(func_ptr, prefix_len),
                                 regno);
    *orig_func_p = orig_func;

    return UCS_OK;
}

void ucm_bistro_patch_lock(void *dst)
{
    static const ucm_bistro_lock_t self_jmp = {
        .b = UCM_AARCH64_B(0),
    };

    ucm_bistro_modify_code(dst, &self_jmp);
}

ucs_status_t ucm_bistro_patch(void *func_ptr, void *hook, const char *symbol,
                              void **orig_func_p,
                              ucm_bistro_restore_point_t **rp)
{
    ucm_bistro_jmp_indirect_t patch;
    ucs_status_t status;

    ucm_bistro_init_jmp_indirect(&patch, hook, 15); /* x15 is caller saved */

    if (orig_func_p != NULL) {
        status = ucm_bistro_relocate_func(func_ptr, sizeof(patch), symbol,
                                          orig_func_p);
        if (status != UCS_OK) {
            return status;
        }
    }

    status = ucm_bistro_create_restore_point(func_ptr, sizeof(patch), rp);
    if (UCS_STATUS_IS_ERR(status)) {
        return status;
    }

    return ucm_bistro_apply_patch_atomic(func_ptr, &patch, sizeof(patch));
}

#endif
