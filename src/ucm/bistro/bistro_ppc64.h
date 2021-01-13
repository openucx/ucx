/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.       ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCM_BISTRO_BISTRO_PPC64_H_
#define UCM_BISTRO_BISTRO_PPC64_H_

#include "bistro.h"

#include <ucs/type/status.h>
#include <stdint.h>
#include <stddef.h>


/* special processing for ppc64 to save and restore TOC (r2)
 * Reference: "64-bit PowerPC ELF Application Binary Interface Supplement 1.9" */
#define UCM_BISTRO_PROLOGUE                       \
    uint64_t toc_save;                            \
    asm volatile ("std 2, %0" : "=m" (toc_save)); \
    asm volatile ("nop; nop; nop; nop; nop");
#define UCM_BISTRO_EPILOGUE \
    asm volatile ("ld  2, %0" : : "m" (toc_save));


/**
 * Set library function call hook using Binary Instrumentation
 * method (BISTRO): replace function body by user defined call
 *
 * @param func_ptr  Function to patch.
 * @param hook      User-defined function-replacer.
 * @param rp        Restore point used to restore original function.
                    Optional, may be NULL.
 *
 * @return Error code as defined by @ref ucs_status_t
 */
/* we have to use inline proxy call to save TOC register
 * value - PPC is very sensible to this register value */
ucs_status_t ucm_bistro_patch_toc(void *func_ptr, void *hook,
                                  ucm_bistro_restore_point_t **rp,
                                  uint64_t toc);

static inline ucs_status_t
ucm_bistro_patch(void *func_ptr, void *hook, const char *symbol,
                 void **orig_func_p, ucm_bistro_restore_point_t **rp)
{
    uint64_t toc;

    if (orig_func_p != NULL) {
        return UCS_ERR_UNSUPPORTED;
    }

    asm volatile ("std 2, %0" : "=m" (toc));
    return ucm_bistro_patch_toc(func_ptr, hook, rp, toc);
}

#endif
