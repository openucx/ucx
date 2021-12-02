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


typedef struct uct_cuda_copy_queue_desc {
    /* stream on which asynchronous memcpy operations are enqueued */
    cudaStream_t                stream;
    /* queue of cuda events */
    ucs_queue_head_t            event_queue;
    /* needed to allow queue descriptor to be added to iface->active_queue */
    ucs_queue_elem_t            queue;
} uct_cuda_copy_queue_desc_t;


typedef struct uct_cuda_copy_iface {
    uct_base_iface_t            super;
    /* used to store uuid and check iface reachability */
    uct_cuda_copy_iface_addr_t  id;
    /* pool of cuda events to check completion of memcpy operations */
    ucs_mpool_t                 cuda_event_desc;
    /* list of queues which require progress */
    ucs_queue_head_t            active_queue;
    /* stream used to issue short operations */
    cudaStream_t                short_stream;
    /* stream used to issue short operations */
    CUcontext                   cuda_context;
    /* array of queue descriptors for each src/dst memory type combination */
    uct_cuda_copy_queue_desc_t  queue_desc[UCS_MEMORY_TYPE_LAST][UCS_MEMORY_TYPE_LAST];
    /* config parameters to control cuda copy transport */
    struct {
        unsigned                max_poll;
        unsigned                max_cuda_events;
        double                  bandwidth;
    } config;
    /* handler to support arm/wakeup feature */
    struct {
        void                    *event_arg;
        uct_async_event_cb_t    event_cb;
    } async;
} uct_cuda_copy_iface_t;


typedef struct uct_cuda_copy_iface_config {
    uct_iface_config_t      super;
    unsigned                max_poll;
    unsigned                max_cuda_events;
    double                  bandwidth;
} uct_cuda_copy_iface_config_t;


typedef struct uct_cuda_copy_event_desc {
    cudaEvent_t      event;
    uct_completion_t *comp;
    ucs_queue_elem_t queue;
} uct_cuda_copy_event_desc_t;
#endif
