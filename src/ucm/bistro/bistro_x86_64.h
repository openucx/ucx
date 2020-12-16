/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.       ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCM_BISTRO_BISTRO_X86_64_H_
#define UCM_BISTRO_BISTRO_X86_64_H_

#include <stdint.h>

#include <ucs/type/status.h>
#include <ucs/sys/compiler_def.h>

#define UCM_BISTRO_PROLOGUE
#define UCM_BISTRO_EPILOGUE

/* Patch by jumping to absolute address loaded from register */
typedef struct ucm_bistro_jmp_r11_patch {
    uint8_t mov_r11[2];  /* mov %r11, addr */
    void    *ptr;
    uint8_t jmp_r11[3];  /* jmp r11        */
} UCS_S_PACKED ucm_bistro_jmp_r11_patch_t;


/* Patch by jumping to relative address by immediate displacement */
typedef struct ucm_bistro_jmp_near_patch {
    uint8_t jmp_rel; /* opcode:  JMP rel32          */
    int32_t disp;    /* operand: jump displacement */
} UCS_S_PACKED ucm_bistro_jmp_near_patch_t;


/**
 * Set library function call hook using Binary Instrumentation
 * method (BISTRO): replace function body by user defined call
 *
 * @param func_ptr     Pointer to function to patch.
 * @param hook         User-defined function-replacer.
 * @param symbol       Function name to replace.
 * @param orig_func_p  If non-NULL, set to pointer to a trampoline which calls
 *                     the original function (before patching). If it's not
 *                     possible to create such trampoline, the function returns
 *                     @ref UCS_ERR_UNSUPPORTED.
 * @param rp           Restore point used to restore original function.
 *                     Optional, may be NULL.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucm_bistro_patch(void *func_ptr, void *hook, const char *symbol,
                              void **orig_func_p,
                              ucm_bistro_restore_point_t **rp);

#endif
