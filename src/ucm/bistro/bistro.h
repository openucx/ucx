/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.       ALL RIGHTS RESERVED.
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
#else
#  error "Unsupported architecture"
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
ucs_status_t ucm_bistro_restore(ucm_bistro_restore_point_t *rp);

/**
 * Remove resore point created by @ref ucm_bistro_patch witout
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

#endif
