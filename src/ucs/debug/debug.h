/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2021.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_DEBUG_H_
#define UCS_DEBUG_H_

#include <ucs/sys/compiler_def.h>

BEGIN_C_DECLS

/**
 * Disable signal handling in UCS for signal.
 * Previous signal handler is set.
 * @param signum   Signal number to disable handling.
 */
void ucs_debug_disable_signal(int signum);

END_C_DECLS

#endif
