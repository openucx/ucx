/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2019. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_STUBS_H
#define UCS_STUBS_H

#include <ucs/type/status.h>

#include <sys/types.h>
#include <stdlib.h>
#include <stdint.h>

BEGIN_C_DECLS

/** @file stubs.h */

/**
 * Empty function which can be casted to a no-operation callback in various situations.
 */
void ucs_empty_function(void);
unsigned ucs_empty_function_return_zero(void);
unsigned ucs_empty_function_return_one(void);
int ucs_empty_function_return_one_int(void);
int64_t ucs_empty_function_return_zero_int64(void);
int ucs_empty_function_return_zero_int(void);
size_t ucs_empty_function_return_zero_size_t(void);
ucs_status_t ucs_empty_function_return_success(void);
ucs_status_t ucs_empty_function_return_unsupported(void);
ucs_status_ptr_t ucs_empty_function_return_ptr_unsupported(void);
ucs_status_t ucs_empty_function_return_inprogress(void);
ucs_status_t ucs_empty_function_return_no_resource(void);
ucs_status_t ucs_empty_function_return_invalid_param(void);
ucs_status_ptr_t ucs_empty_function_return_ptr_no_resource(void);
ucs_status_t ucs_empty_function_return_ep_timeout(void);
ssize_t ucs_empty_function_return_bc_ep_timeout(void);
ucs_status_t ucs_empty_function_return_busy(void);
int ucs_empty_function_do_assert(void);
void ucs_empty_function_do_assert_void(void);

END_C_DECLS

#endif
