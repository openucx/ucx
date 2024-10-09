/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2018. ALL RIGHTS RESERVED.
 * Copyright (C) Tactical Computing Labs, LLC. 2022. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCM_BISTRO_BISTRO_H_
#define UCM_BISTRO_BISTRO_H_

#include <stdint.h>

#include <ucs/type/status.h>

typedef struct ucm_bistro_restore_point ucm_bistro_restore_point_t;

#if defined(__powerpc64__)
#  include "bistro_ppc64.h"
#elif defined(__aarch64__)
#  include "bistro_aarch64.h"
#elif defined(__x86_64__)
#  include "bistro_x86_64.h"
#elif defined(__riscv)
#  include "bistro_rv64.h"
#else
#  error "Unsupported architecture"
#endif


/* Context for copying code to another location while preserving its previous
   functionality */
typedef struct {
    const void *src_p;   /* Pointer to current source instruction */
    const void *src_end; /* Upper limit for source instructions */
    void       *dst_p;   /* Pointer to current destination instruction */
    void       *dst_end; /* Upper limit for destination instructions */
} ucm_bistro_relocate_context_t;


/**
 * Restore original function body using restore point created
 * by @ref ucm_bistro_patch
 *
 * @param rp     restore point, is removed after success operation
 *               completed
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucm_bistro_restore(ucm_bistro_restore_point_t *rp);


/**
 * Remove restore point created by @ref ucm_bistro_patch without
 * restore original function body
 *
 * @param rp     restore point
 *
 * @return Error code as defined by @ref ucs_status_t
 */
ucs_status_t ucm_bistro_remove_restore_point(ucm_bistro_restore_point_t *rp);


/**
 * Get patch address for restore point
 *
 * @param rp     restore point
 *
 * @return Address of patched function body
 */
void *ucm_bistro_restore_addr(ucm_bistro_restore_point_t *rp);


/**
 * Allocate executable memory which can be used to create trampolines or
 * temporary functions.
 *
 * @param size   Memory size to allocated
 *
 * @return Pointer to allocated memory, or NULL if failed.
 */
void *ucm_bistro_allocate_code(size_t size);


/**
 * Relocate a single instruction to new address. The implementation is
 * architecture-specific.
 *
 * @param ctx Instruction relocation context
 *
 * @return ucs_status_t UCS_OK if successful, otherwise an error code.
 */
ucs_status_t ucm_bistro_relocate_one(ucm_bistro_relocate_context_t *ctx);


/*
 * Relocate at least 'min_src_length' code instructions from 'src' to 'dst',
 * possibly changing some of them to new instructions.
 * Uses a  simplified disassembler which supports only typical instructions
 * found in function prologue.
 */
ucs_status_t
ucm_bistro_relocate_code(void *dst, const void *src, size_t min_src_length,
                         size_t max_dst_length, size_t *dst_length_p,
                         size_t *src_length_p, const char *symbol,
                         ucm_bistro_relocate_context_t *ctx);

#endif
