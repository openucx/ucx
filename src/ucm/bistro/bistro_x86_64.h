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

typedef struct ucm_bistro_patch {
    uint8_t mov_r11[2];  /* mov %r11, addr */
    void    *ptr;
    uint8_t jmp_r11[3];  /* jmp r11        */
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
