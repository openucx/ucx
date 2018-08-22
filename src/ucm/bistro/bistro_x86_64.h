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
