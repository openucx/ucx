/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.       ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCM_BISTRO_BISTRO_AARCH64_H_
#define UCM_BISTRO_BISTRO_AARCH64_H_

#include <stdint.h>

#include <ucs/type/status.h>
#include <ucs/sys/compiler_def.h>

#define UCM_BISTRO_PROLOGUE
#define UCM_BISTRO_EPILOGUE

typedef struct ucm_bistro_patch {
    uint32_t reg3;  /* movz    x15, addr, lsl #48 */
    uint32_t reg2;  /* movk    x15, addr, lsl #32 */
    uint32_t reg1;  /* movk    x15, addr, lsl #16 */
    uint32_t reg0;  /* movk    x15, addr          */
    uint32_t br;    /* br      x15                */
} UCS_S_PACKED ucm_bistro_patch_t;

/**
 * Set library function call hook using Binary Instrumentation
 * method (BISTRO): replace function body by user defined call
 *
 * @param symbol function name to replace
 * @param hook   user-defined function-replacer
 * @param rp     restore point used to restore original function,
 *               optional, may be NULL
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucm_bistro_patch(const char *symbol, void *hook,
                              ucm_bistro_restore_point_t **rp);

#endif
