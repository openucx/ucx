/**
 * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_COPY_IFACE_H
#define UCT_CUDA_COPY_IFACE_H

#include <uct/base/uct_iface.h>
#include <uct/cuda/base/cuda_iface.h>
#include <ucs/memory/memory_type.h>
#include <pthread.h>


typedef uint64_t uct_cuda_copy_iface_addr_t;


typedef struct uct_cuda_copy_iface {
    uct_base_iface_t            super;
    uct_cuda_copy_iface_addr_t  id;
    ucs_mpool_t                 cuda_completion_desc;
    ucs_queue_head_t            outstanding_q;
    cudaStream_t                stream_short_ops; /* stream for short operations */
    struct {
        unsigned                max_poll;
        unsigned                max_entries;
        unsigned                num_descs;
    } config;
    struct {
        void                    *event_arg;
        uct_async_event_cb_t    event_cb;
    } async;
} uct_cuda_copy_iface_t;


typedef struct uct_cuda_copy_iface_config {
    uct_iface_config_t      super;
    unsigned                max_poll;
    unsigned                max_entries;
    unsigned                num_descs;
} uct_cuda_copy_iface_config_t;


typedef struct uct_cuda_copy_completion_desc {
    cudaEvent_t event;
    cudaStream_t stream;
    uct_completion_t *comp;
    ucs_queue_elem_t  queue;
} uct_cuda_copy_completion_desc_t;

#endif
