/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2017. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_COPY_IFACE_H
#define UCT_CUDA_COPY_IFACE_H


#include <ucs/datastruct/static_bitmap.h>
#include <ucs/memory/memory_type.h>
#include <uct/base/uct_iface.h>
#include <uct/cuda/base/cuda_iface.h>

#include <pthread.h>


#define UCT_CUDA_MEMORY_TYPES_MAP 64

typedef uint64_t uct_cuda_copy_iface_addr_t;


/*
    uct_cu_stream_bitmap_t will be treated as a 2D bitmap, in which
    each bit represents a CUstream from the queue_desc attr:
    row index is source mem_type and column index is the dest mem_type.

    For example:
    H - Host, C - Cuda, R - ROCm, I - Infiniband (RDMA)

      H C R I
    H 0 0 0 0 
    C 0 0 0 0 
    R 0 0 0 0 
    I 0 0 0 0

    Bits will be set using:
    UCS_BITMAP_SET(bitmap, uct_cuda_copy_flush_bitmap_idx(src_mem_type, dst_mem_type))
*/
typedef ucs_static_bitmap_s(UCT_CUDA_MEMORY_TYPES_MAP) uct_cu_stream_bitmap_t;


typedef struct uct_cuda_copy_bw {
    double            h2d;
    double            d2h;
    double            d2d;
    double            dflt;
} uct_cuda_copy_bw_t;


typedef struct {
    uct_cuda_ctx_rsc_t    super;
    /* stream used to issue short operations */
    CUstream              short_stream;
    /* array of queue descriptors for each src/dst memory type combination */
    uct_cuda_queue_desc_t queue_desc[UCS_MEMORY_TYPE_LAST]
                                    [UCS_MEMORY_TYPE_LAST];
} uct_cuda_copy_ctx_rsc_t;


typedef struct uct_cuda_copy_iface {
    uct_cuda_iface_t            super;
    /* used to store uuid and check iface reachability */
    uct_cuda_copy_iface_addr_t  id;
    /* config parameters to control cuda copy transport */
    struct {
        uct_cuda_copy_bw_t      bw;
    } config;
    /* handler to support arm/wakeup feature */
    struct {
        void                    *event_arg;
        uct_async_event_cb_t    event_cb;
    } async;

    /* 2D bitmap representing which streams in queue_desc matrix 
       should sync during flush */
    uct_cu_stream_bitmap_t streams_to_sync;
} uct_cuda_copy_iface_t;


typedef struct uct_cuda_copy_iface_config {
    uct_iface_config_t      super;
    unsigned                max_poll;
    unsigned                max_cuda_events;
    uct_cuda_copy_bw_t      bw;
} uct_cuda_copy_iface_config_t;


static UCS_F_ALWAYS_INLINE unsigned
uct_cuda_copy_flush_bitmap_idx(ucs_memory_type_t src_mem_type,
                               ucs_memory_type_t dst_mem_type)
{
    return (src_mem_type * UCS_MEMORY_TYPE_LAST) + dst_mem_type;
}

#endif
