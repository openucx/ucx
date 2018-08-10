/**
 * Copyright (C) Mellanox Technologies Ltd. 2018        ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCM_UTIL_BISTRO_H_
#define UCM_UTIL_BISTRO_H_

#include <stdint.h>

#include <ucs/type/status.h>

typedef struct ucm_bistro_restore_point ucm_bistro_restore_point_t;
typedef ucm_bistro_restore_point_t *ucm_bistro_restore_point_h;

#if defined(__powerpc64__)
/* special processing for ppc64 to save and restore TOC (r2)
 * Reference: "64-bit PowerPC ELF Application Binary Interface Supplement 1.9" */
#  define UCM_BISTRO_PROLOGUE \
    uint64_t toc_save; \
    asm volatile ("std 2, %0" : "=m" (toc_save)); \
    asm volatile ("nop; nop; nop; nop; nop");
#  define UCM_BISTRO_EPILOGUE \
    asm volatile ("ld  2, %0" : : "m" (toc_save));
#else /* X86 or ARM */
#  define UCM_BISTRO_PROLOGUE
#  define UCM_BISTRO_EPILOGUE
#endif


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
#if !defined (__powerpc64__)
ucs_status_t ucm_bistro_patch(const char *symbol, void *hook,
                              ucm_bistro_restore_point_h *rp);
#else
/* we have to use inline proxy call to save TOC register
 * value - PPC is very sensible to this register value */
ucs_status_t ucm_bistro_patch_toc(const char *symbol, void *hook,
                                  ucm_bistro_restore_point_h *rp,
                                  uint64_t toc);

static inline
ucs_status_t ucm_bistro_patch(const char *symbol, void *hook,
                              ucm_bistro_restore_point_h *rp)
{
    uint64_t toc;
    asm volatile ("std 2, %0" : "=m" (toc));
    return ucm_bistro_patch_toc(symbol, hook, rp, toc);
}
#endif

/**
 * Restore original function body using restore point created
 * by @ref ucm_bistro_patch
 *
 * @param rp     restore point, is removed after success operation
 *               completed
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucm_bistro_restore(ucm_bistro_restore_point_h rp);

/**
 * Remove resore point created by @ref ucm_bistro_patch witout
 * restore original function body
 *
 * @param rp     restore point
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucm_bistro_remove_restore_point(ucm_bistro_restore_point_h rp);

/**
 * Get patch address for restore point
 *
 * @param rp     restore point
 *
 * @return Address of patched function body
 */
void *ucm_bistro_restore_addr(ucm_bistro_restore_point_h rp);

#endif
