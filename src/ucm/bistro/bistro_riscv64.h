/**
 * Copyright (C) Tactical Computing Labs, LLC. 2022. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCM_BISTRO_BISTRO_RISCV64_H_
#define UCM_BISTRO_BISTRO_RISCV64_H_

#include <stdint.h>

#include <ucs/type/status.h>
#include <ucs/sys/compiler_def.h>

#define UCM_BISTRO_PROLOGUE
#define UCM_BISTRO_EPILOGUE

typedef struct ucm_bistro_patch {
    uint32_t reg14;   /* addi x31, x0, (addr>>52)                   */
    uint32_t reg13;   /* slli x31, x31, 52                          */
    uint32_t reg12;   /* addi x30, x0, ((addr>>40)&0b111111111111)  */
    uint32_t reg11;   /* slli x30, x30, 40                          */
    uint32_t reg10;   /* or   x31, x31, X30                         */
    uint32_t reg9;    /* addi x29, x0, ((addr>>>28)&0b111111111111) */
    uint32_t reg8;    /* slli x29, x29, 28                          */
    uint32_t reg7;    /* or   x31, x31, x29                         */
    uint32_t reg6;    /* addi x28, x0, ((addr>>>16)&0b111111111111) */
    uint32_t reg5;    /* slli x28, x28, 16                          */
    uint32_t reg4;    /* or   x31, x31, x28                         */
    uint32_t reg3;    /* addi x30, x0, ((addr>>>4)&0b111111111111) */
    uint32_t reg2;    /* slli x30, x30, 4                           */
    uint32_t reg1;    /* or   x31, x31, x30                         */
    uint32_t reg0;    /* ori  x31, x31, (addr&0b1111)               */
    uint32_t br;      /* br x31 */
} UCS_S_PACKED ucm_bistro_patch_t;

/**
 * Set library function call hook using Binary Instrumentation
 * method (BISTRO): replace function body by user defined call
 *
 * @param func_ptr     Pointer to function to patch.
 * @param hook         User-defined function-replacer.
 * @param symbol       Function name to replace.
 * @param orig_func_p  Unsupported on this architecture and must be NULL.
 *                     If set to a non-NULL value, this function returns
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
