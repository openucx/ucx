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


/*
 * Find the minimal length of initial instructions in the function which can be
 * safely executed from any memory location.
 * Uses a very simplified disassembler which supports only the typical
 * instructions found in function prologue.
 */
static size_t ucm_bistro_detect_pic_prefix(const void *func, size_t min_length)
{
    size_t offset, prev_offset;
    uint8_t rex, opcode, modrm;

    offset = 0;
    while (offset < min_length) {
        prev_offset = offset;
        opcode      = *(uint8_t*)UCS_PTR_BYTE_OFFSET(func, offset++);

        /* check for REX prefix */
        if ((opcode & 0xF0) == 0x40) {
            rex    = opcode;
            opcode = *(uint8_t*)UCS_PTR_BYTE_OFFSET(func, offset++);
        } else {
            rex = 0;
        }

        /* check the opcode */
        if (((rex == 0) || rex == 0x41) && ((opcode & 0xF0) == 0x50)) {
            /* push <register> */
            continue;
        } else if ((rex == 0x48) && (opcode == 0x81)) {
            /* group 1A operation */
            modrm = *(uint8_t*)UCS_PTR_BYTE_OFFSET(func, offset++);
            if (modrm == 0xEC) {
                /* sub $imm32, %rsp */
                offset += sizeof(uint32_t);
                continue;
            }
        } else if ((rex == 0x48) && (opcode == 0x89)) {
            /* mov %rsp, %rbp */
            modrm = *(uint8_t*)UCS_PTR_BYTE_OFFSET(func, offset++);
            if (modrm == 0xE5) {
                continue;
            }
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
