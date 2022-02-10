/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.       ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

/* *******************************************************
 * x86 processors family                                 *
 * ***************************************************** */

#if defined(__x86_64__)

#include <sys/mman.h>
#include <string.h>
#include <stdlib.h>

#include <ucm/bistro/bistro.h>
#include <ucm/bistro/bistro_int.h>
#include <ucm/util/sys.h>
#include <ucs/sys/math.h>
#include <ucs/type/serialize.h>


typedef struct {
    void *jmp_addr;
    char code[];
} ucm_bistro_orig_func_t;

/* Patch by jumping to absolute address loaded from register */
typedef struct ucm_bistro_jmp_rax_patch {
    uint8_t mov_rax[2];  /* mov %rax, addr */
    void    *ptr;
    uint8_t jmp_rax[2];  /* jmp rax        */
} UCS_S_PACKED ucm_bistro_jmp_rax_patch_t;

/* Patch by jumping to relative address by immediate displacement */
typedef struct ucm_bistro_jmp_near_patch {
    uint8_t jmp_rel; /* opcode:  JMP rel32          */
    int32_t disp;    /* operand: jump displacement */
} UCS_S_PACKED ucm_bistro_jmp_near_patch_t;

typedef struct {
    uint8_t opcode; /* 0xff */
    uint8_t modrm; /* 0x25 */
    int32_t displ;
} UCS_S_PACKED ucm_bistro_jmp_indirect_t;

typedef struct {
    uint8_t  push_rax;
    uint8_t  movabs_rax[2];
    uint64_t rax_value;
    uint8_t  cmp_dptr_rax[2];
    uint32_t cmp_value;
    uint8_t  pop_rax;
} UCS_S_PACKED ucm_bistro_cmp_xlt_t;

typedef struct {
    uint8_t jmp_rel[2];
    uint8_t jmp_out[2];
    struct {
        uint8_t  push_imm;
        uint32_t value;
    } UCS_S_PACKED hi, lo;
    uint8_t        ret;
} UCS_S_PACKED ucm_bistro_jcc_xlt_t;

typedef struct {
    const void *src_p;   /* Pointer to current source instruction */
    const void *src_end; /* Upper limit for source instructions */
    void       *dst_p;   /* Pointer to current destination instruction */
    void       *dst_end; /* Upper limit for destination instructions */
} ucm_bistro_relocate_context_t;


/* REX prefix */
#define UCM_BISTRO_X86_REX_MASK  0xF0 /* Mask */
#define UCM_BISTRO_X86_REX       0x40 /* Value */

#define UCM_BISTRO_X86_REX_W     0x48 /* REX.W value */
#define UCM_BISTRO_X86_REX_B     0x41 /* REX.B value */

/* PUSH general register
 * "push $reg"
 */
#define UCM_BISTRO_X86_PUSH_R_MASK 0xF0 /* Mask */
#define UCM_BISTRO_X86_PUSH_R      0x50 /* Value */

/* Immediate Grp 1(1A), Ev, Iz */
#define UCM_BISTRO_X86_IMM_GRP1_EV_IZ 0x81

/* MOV Ev,Gv */
#define UCM_BISTRO_X86_MOV_EV_GV 0x89

/* MOV immediate word or double into word, double, or quad register
 * "mov $imm32, %reg"
 */
#define UCM_BISTRO_X86_MOV_IR_MASK 0xF8 /* Mask */
#define UCM_BISTRO_X86_MOV_IR      0xB8 /* Value */

/* ModR/M encoding:
 * [ mod | reg   | r/m   ]
 * [ 7 6 | 5 4 3 | 2 1 0 ]
 */
#define UCM_BISTRO_X86_MODRM_MOD_SHIFT 6 /* mod */
#define UCM_BISTRO_X86_MODRM_REG_SHIFT 3 /* reg */
#define UCM_BISTRO_X86_MODRM_RM_BITS   3 /* r/m */

/* Table 2-2 */
#define UCM_BISTRO_X86_MODRM_MOD_DISP8  1 /* 0b01 */
#define UCM_BISTRO_X86_MODRM_MOD_DISP32 2 /* 0b10 */
#define UCM_BISTRO_X86_MODRM_MOD_REG    3 /* 0b11 */
#define UCM_BISTRO_X86_MODRM_RM_SIB     4 /* 0b100 */

/* ModR/M encoding for SUB RSP
 * mod=0b11, reg=0b101 (SUB as opcode extension), r/m=0b100
 */
#define UCM_BISTRO_X86_MODRM_SUB_SP 0xEC /* 11 101 100 */

/* ModR/M encoding for EBP/BP/CH/MM5/XMM5, AH/SP/ESP/MM4/XMM4 */
#define UCM_BISTRO_X86_MODRM_BP_SP 0xE5 /* 11 100 101 */

/* ModR/M encoding for CMP [RIP+x], Imm32 */
#define UCM_BISTRO_X86_MODRM_CMP_RIP 0x3D /* 11 111 101 */

/* Jcc (conditional jump) opcodes range */
#define UCM_BISTRO_X86_JCC_FIRST 0x70
#define UCM_BISTRO_X86_JCC_LAST  0x7F


static ucs_status_t
ucm_bistro_relocate_one(ucm_bistro_relocate_context_t *ctx)
{
    const void *copy_src     = ctx->src_p;
    ucm_bistro_cmp_xlt_t cmp = {
        .push_rax     = 0x50,
        .movabs_rax   = {0x48, 0xb8},
        .cmp_dptr_rax = {0x81, 0x38},
        .pop_rax      = 0x58
    };
    ucm_bistro_jcc_xlt_t jcc = {
        .jmp_rel = {0x00, 0x02},
        .jmp_out = {0xeb, 0x0b},
        .hi      = {0x68, 0},
        .lo      = {0x68, 0},
        .ret     = 0xc3
    };
    uint8_t rex, opcode, modrm, mod;
    size_t dst_length;
    uint64_t jmpdest;
    int32_t disp32;
    uint32_t imm32;
    int8_t disp8;

    /* Check opcode and REX prefix */
    opcode = *ucs_serialize_next(&ctx->src_p, const uint8_t);
    if ((opcode & UCM_BISTRO_X86_REX_MASK) == UCM_BISTRO_X86_REX) {
        rex    = opcode;
        opcode = *ucs_serialize_next(&ctx->src_p, const uint8_t);
    } else {
        rex = 0;
    }

    if (((rex == 0) || rex == UCM_BISTRO_X86_REX_B) &&
        ((opcode & UCM_BISTRO_X86_PUSH_R_MASK) == UCM_BISTRO_X86_PUSH_R)) {
        /* push reg */
        goto out_copy_src;
    } else if ((rex == UCM_BISTRO_X86_REX_W) &&
               (opcode == UCM_BISTRO_X86_IMM_GRP1_EV_IZ)) {
        modrm = *ucs_serialize_next(&ctx->src_p, const uint8_t);
        if (modrm == UCM_BISTRO_X86_MODRM_SUB_SP) {
            /* sub $imm32, %rsp */
            ucs_serialize_next(&ctx->src_p, const uint32_t);
            goto out_copy_src;
        }
    } else if ((rex == UCM_BISTRO_X86_REX_W) &&
               (opcode == UCM_BISTRO_X86_MOV_EV_GV)) {
        modrm = *ucs_serialize_next(&ctx->src_p, const uint8_t);
        mod   = modrm >> UCM_BISTRO_X86_MODRM_MOD_SHIFT;
        if (modrm == UCM_BISTRO_X86_MODRM_BP_SP) {
            /* mov %rsp, %rbp */
            goto out_copy_src;
        }

        if ((mod != UCM_BISTRO_X86_MODRM_MOD_REG) &&
            ((modrm & UCS_MASK(UCM_BISTRO_X86_MODRM_RM_BITS)) ==
             UCM_BISTRO_X86_MODRM_RM_SIB)) {
            /* r/m = 0b100, mod = 0b00/0b01/0b10 */
            ucs_serialize_next(&ctx->src_p, const uint8_t); /* skip SIB */
            if (mod == UCM_BISTRO_X86_MODRM_MOD_DISP8) {
                ucs_serialize_next(&ctx->src_p, const uint8_t); /* skip disp8 */
                goto out_copy_src;
            } else if (mod == UCM_BISTRO_X86_MODRM_MOD_DISP32) {
                ucs_serialize_next(&ctx->src_p, const uint32_t); /* skip disp32 */
                goto out_copy_src;
            }
        }
    } else if ((rex == 0) && ((opcode & UCM_BISTRO_X86_MOV_IR_MASK) ==
                              UCM_BISTRO_X86_MOV_IR)) {
        /* mov $imm32, %reg */
        ucs_serialize_next(&ctx->src_p, const uint32_t);
        goto out_copy_src;
    } else if ((rex == 0) && (opcode == UCM_BISTRO_X86_IMM_GRP1_EV_IZ)) {
        modrm = *ucs_serialize_next(&ctx->src_p, const uint8_t);
        if (modrm == UCM_BISTRO_X86_MODRM_CMP_RIP) {
            /*
             * Since we can't assume the new code will be within 32-bit
             * range of the global variable argument, we need to translate
             * the code from:
             *   cmpl $imm32, $disp32(%rip)
             * to:
             *   push %rax
             *   movq $addr64, %rax ; $addr64 is $disp32+%rip
             *   cmpl $imm32, (%rax)
             *   pop %rax
             */
            disp32        = *ucs_serialize_next(&ctx->src_p, const int32_t);
            imm32         = *ucs_serialize_next(&ctx->src_p, const uint32_t);
            cmp.rax_value = (uintptr_t)UCS_PTR_BYTE_OFFSET(ctx->src_p, disp32);
            cmp.cmp_value = imm32;
            copy_src      = &cmp;
            dst_length    = sizeof(cmp);
            goto out_copy;
        }
    } else if ((rex == 0) && (opcode >= UCM_BISTRO_X86_JCC_FIRST) &&
               (opcode <= UCM_BISTRO_X86_JCC_LAST)) {
        /*
         * Since we can't assume the new code will be within 32-bit range of the
         * jump destination, we need to translate the code from:
         *        jCC $disp8
         * to:
         *        jCC L1
         *    L1: jmp L2        ; condition 'CC' did not hold
         *        push $addrhi
         *        push $addrlo
         *        ret           ; 64-bit jump to destination
         *    L2:               ; continue execution
         */
        disp8          = *ucs_serialize_next(&ctx->src_p, const int8_t);
        jmpdest        = (uintptr_t)UCS_PTR_BYTE_OFFSET(ctx->src_p, disp8);
        jcc.jmp_rel[0] = opcode; /* keep original jump condition */
        jcc.hi.value   = jmpdest >> 32;
        jcc.lo.value   = jmpdest & UCS_MASK(32);
        copy_src       = &jcc;
        dst_length     = sizeof(jcc);
        /* Prevent patching past jump target */
        ctx->src_end   = ucs_min(ctx->src_end, (void*)jmpdest);
        goto out_copy;
    }

    /* Could not recognize the instruction */
    return UCS_ERR_UNSUPPORTED;

out_copy_src:
    dst_length = UCS_PTR_BYTE_DIFF(copy_src, ctx->src_p);
out_copy:
    if (UCS_PTR_BYTE_OFFSET(ctx->dst_p, dst_length) > ctx->dst_end) {
        return UCS_ERR_BUFFER_TOO_SMALL;
    }

    /* Copy 'dst_length' bytes to ctx->dst_p and advance it */
    memcpy(ucs_serialize_next_raw(&ctx->dst_p, void, dst_length), copy_src,
           dst_length);
    return UCS_OK;
}

/*
 * Relocate at least 'min_src_length' code instructions from 'src' to 'dst',
 * possibly changing some of them to new instructions.
 * Uses a  simplified disassembler which supports only typical instructions
 * found in function prologue.
 */
static ucs_status_t
ucm_bistro_relocate_code(void *dst, const void *src, size_t min_src_length,
                         size_t max_dst_length, size_t *dst_length_p,
                         size_t *src_length_p)
{
    ucm_bistro_relocate_context_t ctx = {
        .src_p   = src,
        .dst_p   = dst,
        .dst_end = UCS_PTR_BYTE_OFFSET(dst, max_dst_length),
        .src_end = (void*)UINTPTR_MAX
    };
    ucs_status_t status;

    while (ctx.src_p < UCS_PTR_BYTE_OFFSET(src, min_src_length)) {
        status = ucm_bistro_relocate_one(&ctx);
        if (status != UCS_OK) {
            return status;
        }

        if (ctx.src_p > ctx.src_end) {
            return UCS_ERR_UNSUPPORTED;
        }
    }

    *src_length_p = UCS_PTR_BYTE_DIFF(src, ctx.src_p);
    *dst_length_p = UCS_PTR_BYTE_DIFF(dst, ctx.dst_p);
    return UCS_OK;
}

static const char *
ucm_bistro_dump_code(const void *code, size_t length, char *str, size_t max)
{
    const void *code_p = code;
    char *p            = str;
    char *endp         = str + max;

    while (code_p < UCS_PTR_BYTE_OFFSET(code, length)) {
        snprintf(p, endp - p, " %02X",
                 *ucs_serialize_next(&code_p, const uint8_t));
        p += strlen(p);
    }

    return str;
}

static ucs_status_t
ucm_bistro_construct_orig_func(const void *func_ptr, size_t patch_len,
                               const char *symbol, void **orig_func_p)
{
    size_t code_len, prefix_len, max_code_len;
    ucm_bistro_jmp_indirect_t *jmp_back;
    ucm_bistro_orig_func_t *orig_func;
    ucs_status_t status;
    char code_buf[64];
    int dladdr_ret;
    Dl_info dli;

    /* Allocate executable page */
    max_code_len = ucs_max(patch_len + sizeof(ucm_bistro_cmp_xlt_t) +
                                   sizeof(ucm_bistro_jcc_xlt_t),
                           64);
    orig_func    = ucm_bistro_allocate_code(sizeof(*orig_func) + max_code_len +
                                            sizeof(*jmp_back));
    if (orig_func == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    /* Copy and translate code from 'func_ptr' to 'orig_func->code'.
       'code_len' is the code size at destination buffer, and 'prefix_len' is
       how many bytes were translated from 'func_ptr'. */
    status = ucm_bistro_relocate_code(orig_func->code, func_ptr, patch_len,
                                      max_code_len, &code_len, &prefix_len);
    if (status != UCS_OK) {
        dladdr_ret = dladdr(func_ptr, &dli);
        ucm_diag("failed to patch '%s' from %s length %zu code:%s", symbol,
                 (dladdr_ret != 0) ? dli.dli_fname : "(unknown)", patch_len,
                 ucm_bistro_dump_code(func_ptr, 16, code_buf,
                                      sizeof(code_buf)));
        return UCS_ERR_UNSUPPORTED;
    }

    ucm_debug("'%s' at %p code length %zu/%zu prefix length %zu", symbol,
              func_ptr, code_len, patch_len, prefix_len);

    /* Indirect jump to *orig_func->jmp_address */
    orig_func->jmp_addr = UCS_PTR_BYTE_OFFSET(func_ptr, prefix_len);
    jmp_back            = UCS_PTR_BYTE_OFFSET(orig_func->code, code_len);
    jmp_back->opcode    = 0xff;
    jmp_back->modrm     = 0x25;
    jmp_back->displ     = UCS_PTR_BYTE_DIFF(jmp_back + 1, &orig_func->jmp_addr);
    *orig_func_p        = orig_func->code;

    return UCS_OK;
}

ucs_status_t ucm_bistro_patch(void *func_ptr, void *hook, const char *symbol,
                              void **orig_func_p,
                              ucm_bistro_restore_point_t **rp)
{
    ucm_bistro_jmp_rax_patch_t jmp_rax   = {
        .mov_rax = {0x48, 0xb8},
        .jmp_rax = {0xff, 0xe0}
    };
    ucm_bistro_jmp_near_patch_t jmp_near = {
        .jmp_rel = 0xe9
    };
    void *patch, *jmp_base;
    ucs_status_t status;
    ptrdiff_t jmp_disp;
    size_t patch_len;

    jmp_base = UCS_PTR_BYTE_OFFSET(func_ptr, sizeof(jmp_near));
    jmp_disp = UCS_PTR_BYTE_DIFF(jmp_base, hook);
    if (labs(jmp_disp) < INT32_MAX) {
        /* if 32-bit near jump is possible, use it, since it's a short 5-byte
         * instruction which reduces the chances of racing with other thread
         */
        jmp_near.disp = jmp_disp;
        patch         = &jmp_near;
        patch_len     = sizeof(jmp_near);
    } else {
        jmp_rax.ptr = hook;
        patch       = &jmp_rax;
        patch_len   = sizeof(jmp_rax);
    }

    if (orig_func_p != NULL) {
        status = ucm_bistro_construct_orig_func(func_ptr, patch_len, symbol,
                                                orig_func_p);
        if (status != UCS_OK) {
            return status;
        }
    }

    status = ucm_bistro_create_restore_point(func_ptr, patch_len, rp);
    if (UCS_STATUS_IS_ERR(status)) {
        return status;
    }

    return ucm_bistro_apply_patch(func_ptr, patch, patch_len);
}

#endif
