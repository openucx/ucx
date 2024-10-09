/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2021. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_DEBUG_H_
#define UCS_DEBUG_H_

#include <ucs/sys/compiler_def.h>

#ifdef __SANITIZE_ADDRESS__
#include <sanitizer/asan_interface.h>
#include <ucs/debug/assert.h>
#endif


BEGIN_C_DECLS

/**
 * Disable signal handling in UCS for signal.
 * Previous signal handler is set.
 * @param signum   Signal number to disable handling.
 */
void ucs_debug_disable_signal(int signum);


#ifdef __SANITIZE_ADDRESS__
#define UCS_ASAN_ADDRESS_IS_VALID(_ptr, _size) \
    ucs_assertv(!__asan_region_is_poisoned((void*)(_ptr), _size), "%s: %p", \
                #_ptr, (void*)(_ptr))
#else
#define UCS_ASAN_ADDRESS_IS_VALID(_ptr, _size)
#endif

END_C_DECLS

#endif
