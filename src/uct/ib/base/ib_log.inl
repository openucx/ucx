/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_IB_LOG_INL
#define UCT_IB_LOG_INL

#include "ib_md.h"

#include <ucs/debug/log.h>
#include <ucs/time/time.h>

static UCS_F_ALWAYS_INLINE void
uct_ib_reg_mr_trace(const char *title, const uct_ib_md_t *md,
                    const void *address, size_t length, int dmabuf_fd,
                    size_t dmabuf_offset, uint64_t access_flags,
                    const struct ibv_mr *mr, unsigned long retry,
                    ucs_time_t start_time)
{
    ucs_trace("%s(pd=%p addr=%p len=%zu fd=%d offset=%zu access=0x%" PRIx64 "):"
              " mr=%p lkey=0x%x retry=%lu took %.3f ms",
              title, md->pd, address, length, dmabuf_fd, dmabuf_offset,
              access_flags, mr, mr->lkey, retry,
              ucs_time_to_msec(ucs_get_time() - start_time));
    UCS_STATS_UPDATE_COUNTER(md->stats, UCT_IB_MD_STAT_MEM_REG, +1);
}

#endif
