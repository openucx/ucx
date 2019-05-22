/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_STUBS_H
#define UCS_STUBS_H

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/type/status.h>

#include <stdlib.h>
#include <stdint.h>

BEGIN_C_DECLS

/** @file stubs.h */

/**
 * Empty function which can be casted to a no-operation callback in various situations.
 */
void ucs_empty_function();
unsigned ucs_empty_function_return_zero();
int64_t ucs_empty_function_return_zero_int64();
ucs_status_t ucs_empty_function_return_success();
ucs_status_t ucs_empty_function_return_unsupported();
ucs_status_t ucs_empty_function_return_inprogress();
ucs_status_t ucs_empty_function_return_no_resource();
ucs_status_ptr_t ucs_empty_function_return_ptr_no_resource();
ucs_status_t ucs_empty_function_return_ep_timeout();
ssize_t ucs_empty_function_return_bc_ep_timeout();
ucs_status_t ucs_empty_function_return_busy();
int ucs_empty_function_do_assert();

END_C_DECLS

#endif
