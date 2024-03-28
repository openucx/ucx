/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef LOCKLESS_SYNC_DEF_H_
#define LOCKLESS_SYNC_DEF_H_

#include <ucs/sys/compiler_def.h>

#include <inttypes.h>

typedef struct {
    uint32_t refcount;  /**< Reference count, including +1 if object stored
                             in container */
    uint8_t  flags;     /**< Status flags */
} UCS_S_PACKED ucs_ll_sync_obj_t;

#endif
