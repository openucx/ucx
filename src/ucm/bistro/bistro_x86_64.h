/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.       ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCM_BISTRO_BISTRO_X86_64_H_
#define UCM_BISTRO_BISTRO_X86_64_H_

#include <stdint.h>

#include <ucs/type/status.h>

#define UCM_BISTRO_PROLOGUE
#define UCM_BISTRO_EPILOGUE


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
