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


typedef struct {
    void *jmp_addr;
    char code[];
} ucm_bistro_orig_func_t;

typedef struct {
    uint8_t opcode; /* 0xff */
    uint8_t modrm; /* 0x25 */
    int32_t displ;
} UCS_S_PACKED ucm_bistro_jmp_indirect_t;


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


/*
 * Find the minimal length of initial instructions in the function which can be
 * safely executed from any memory location.
 * Uses a very simplified disassembler which supports only the typical
 * instructions found in function prologue.
 */
static size_t ucm_bistro_detect_pic_prefix(const void *func, size_t min_length)
{
    uint8_t rex, opcode, modrm, mod;
    size_t offset, prev_offset;

    offset = 0;
    while (offset < min_length) {
        prev_offset = offset;
        opcode      = *(uint8_t*)UCS_PTR_BYTE_OFFSET(func, offset++);

        /* check for REX prefix */
        if ((opcode & UCM_BISTRO_X86_REX_MASK) == UCM_BISTRO_X86_REX) {
            rex    = opcode;
            opcode = *(uint8_t*)UCS_PTR_BYTE_OFFSET(func, offset++);
        } else {
            rex = 0;
        }

        /* check the opcode */
        if (((rex == 0) || rex == UCM_BISTRO_X86_REX_B) &&
            ((opcode & UCM_BISTRO_X86_PUSH_R_MASK) == UCM_BISTRO_X86_PUSH_R)) {
            continue;
        } else if ((rex == UCM_BISTRO_X86_REX_W) &&
                   (opcode == UCM_BISTRO_X86_IMM_GRP1_EV_IZ)) {
            modrm = *(uint8_t*)UCS_PTR_BYTE_OFFSET(func, offset++);
            if (modrm == UCM_BISTRO_X86_MODRM_SUB_SP) {
                /* sub $imm32, %rsp */
                offset += sizeof(uint32_t);
                continue;
            }
        } else if ((rex == UCM_BISTRO_X86_REX_W) &&
                   (opcode == UCM_BISTRO_X86_MOV_EV_GV)) {
            modrm = *(uint8_t*)UCS_PTR_BYTE_OFFSET(func, offset++);
            if (modrm == UCM_BISTRO_X86_MODRM_BP_SP) {
                /* mov %rsp, %rbp */
                continue;
            }
            mod = modrm >> UCM_BISTRO_X86_MODRM_MOD_SHIFT;
            if ((mod != UCM_BISTRO_X86_MODRM_MOD_REG) &&
                ((modrm & UCS_MASK(UCM_BISTRO_X86_MODRM_RM_BITS)) ==
                 UCM_BISTRO_X86_MODRM_RM_SIB)) {
                /* r/m = 0b100, mod = 0b00/0b01/0b10 */
                ++offset; /* skip SIB */
                if (mod == UCM_BISTRO_X86_MODRM_MOD_DISP8) {
                    offset += sizeof(uint8_t); /* skip disp8 */
                } else if (mod == UCM_BISTRO_X86_MODRM_MOD_DISP32) {
                    offset += sizeof(uint32_t); /* skip disp32 */
                }
                continue;
            }
        } else if ((rex == 0) &&
                   ((opcode & UCM_BISTRO_X86_MOV_IR_MASK) == UCM_BISTRO_X86_MOV_IR)) {
            offset += sizeof(uint32_t);
            continue;
        }

        /* unsupported instruction - bail */
        return prev_offset;
    }

    return offset;
}

static ucs_status_t
ucm_bistro_construct_orig_func(const void *func_ptr, size_t patch_len,
                               const char *symbol, void **orig_func_p)
{
    ucm_bistro_jmp_indirect_t *jmp_back;
    ucm_bistro_orig_func_t *orig_func;
    size_t prefix_len, code_size;

    prefix_len = ucm_bistro_detect_pic_prefix(func_ptr, patch_len);
    ucm_debug("'%s' at %p prefix length %zu/%zu", symbol, func_ptr, prefix_len,
              patch_len);
    if (prefix_len < patch_len) {
        return UCS_ERR_UNSUPPORTED;
    }

    /* Allocate executable page */
    code_size = sizeof(*orig_func) + patch_len + sizeof(*jmp_back);
    orig_func = ucm_bistro_allocate_code(code_size);
    if (orig_func == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    /* Copy code fragment from original function */
    memcpy(orig_func->code, func_ptr, prefix_len);

    /* Indirect jump to *orig_func->jmp_address */
    orig_func->jmp_addr = UCS_PTR_BYTE_OFFSET(func_ptr, prefix_len);
    jmp_back            = UCS_PTR_BYTE_OFFSET(orig_func->code, prefix_len);
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
    ucm_bistro_jmp_r11_patch_t jmp_r11   = {
        .mov_r11 = {0x49, 0xbb},
        .jmp_r11 = {0x41, 0xff, 0xe3}
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
        jmp_r11.ptr = hook;
        patch       = &jmp_r11;
        patch_len   = sizeof(jmp_r11);
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
