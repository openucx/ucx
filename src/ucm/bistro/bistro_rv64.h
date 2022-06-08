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
     uint32_t rega;              /* load bits 63-43          */
     uint32_t regb;              /* add bits 43-31           */
     uint32_t regc;              /* load bits 30-11          */
     uint32_t regd;              /* shift upper 32 bits left */
     uint32_t rege;              /* add bits 10-0            */
     uint32_t regf;              /* perform jump             */
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
