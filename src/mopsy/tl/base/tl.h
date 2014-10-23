/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef MOPSY_TL_H_
#define MOPSY_TL_H_


#include "tl_base.h"

#include <services/sys/error.h>


/**
 * @ingroup CONTEXT
 * @brief Initialize global context.
 *
 * @param [out] context_p   Filled with context handle.
 *
 * @return Error code.
 */
mopsy_status_t mopsy_tl_init(mopsy_tl_context_h *context_p);


/**
 * @ingroup CONTEXT
 * @brief Destroy global context.
 *
 * @param [in] context   Handle to context.
 */
void mopsy_tl_cleanup(mopsy_tl_context_h context);


#endif
