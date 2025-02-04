/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2021. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_DEBUG_H_
#define UCS_DEBUG_H_

#include <ucs/sys/compiler_def.h>
#include <stddef.h>

BEGIN_C_DECLS

/**
 * Disable signal handling in UCS for signal.
 * Previous signal handler is set.
 * @param signum   Signal number to disable handling.
 */
void ucs_debug_disable_signal(int signum);

void ucs_debug_asan_validate_address(const char *ptr_name, void *address,
                                     size_t size);

#ifdef __SANITIZE_ADDRESS__
#define UCS_ASAN_ADDRESS_IS_VALID(_ptr, _size) \
    ucs_debug_asan_validate_address(#_ptr, (void*)(_ptr), (_size))
#else
#define UCS_ASAN_ADDRESS_IS_VALID(_ptr, _size)
#endif

END_C_DECLS

#endif
