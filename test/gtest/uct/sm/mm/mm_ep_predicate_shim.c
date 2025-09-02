/**
* Copyright (C) Advanced Micro Devices, Inc. 2025. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/
#include <stdint.h>

/*
 * Test-only shim that exposes the macro UCT_MM_EP_IS_ABLE_TO_SEND
 * as a callable symbol.
 */
#include "uct/sm/mm/base/mm_ep.c"

int ucx_mm_ep_can_send(uint64_t head, uint64_t tail, unsigned fifo_size)
{
    return UCT_MM_EP_IS_ABLE_TO_SEND(head, tail, fifo_size);
}
