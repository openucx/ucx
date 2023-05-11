/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2023. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_CALLBACKQ_COMPAT_H
#define UCS_CALLBACKQ_COMPAT_H

#include <ucs/sys/compiler_def.h>


/**
 * @ingroup UCS_RESOURCE
 * Callback backward compatibility flags
 */
enum ucs_callbackq_flags {
    UCS_CALLBACKQ_FLAG_FAST    = UCS_BIT(0), /**< Fast-path (best effort) */
    UCS_CALLBACKQ_FLAG_ONESHOT = UCS_BIT(1)  /**< Call the callback only once
                                                  (cannot be used with FAST) */
};

#endif
