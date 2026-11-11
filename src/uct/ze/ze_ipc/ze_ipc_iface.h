/*
 * Copyright (C) Intel Corporation, 2023-2024. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_ZE_IPC_IFACE_H
#define UCT_ZE_IPC_IFACE_H

#include <uct/base/uct_iface.h>
#include <ucs/arch/cpu.h>
#include <ucs/type/spinlock.h>
#include <level_zero/ze_api.h>

#include "ze_ipc_md.h"


#define UCT_ZE_IPC_TL_NAME   "ze_ipc"
#define UCT_ZE_IPC_MAX_PEERS 16


typedef struct uct_ze_ipc_iface_config {
    uct_iface_config_t super;
    unsigned           max_poll; /* query attempts w.o success */
    unsigned max_cmd_lists; /* max number of command lists for parallel progress */
    int      enable_cache; /* enable/disable ipc handle cache */
    unsigned event_pool_size; /* number of events in the shared event pool */
    double bandwidth; /* estimated bandwidth */
    double latency; /* estimated latency */
    double overhead; /* estimated CPU overhead */
} uct_ze_ipc_iface_config_t;


/**
 * Queue descriptor for each command list
 * Similar to CUDA IPC's uct_cuda_queue_desc_t design
 */
typedef struct uct_ze_ipc_queue_desc {
    ucs_queue_elem_t         queue; /* element in active queue list */
    ze_command_list_handle_t cmd_list; /* immediate command list */
    ucs_queue_head_t         event_queue; /* queue of outstanding events */
} uct_ze_ipc_queue_desc_t;


typedef struct uct_ze_ipc_iface {
    uct_base_iface_t          super;
    ze_context_handle_t       ze_context;
    ze_device_handle_t        ze_device;
    uct_ze_ipc_iface_config_t config;

    /* Multi-command-list support for parallel progress */
    uct_ze_ipc_queue_desc_t
            queue_desc[UCT_ZE_IPC_MAX_PEERS]; /* array of queue descriptors */
    ucs_queue_head_t active_queue; /* queue of active queue descriptors */
    unsigned         num_cmd_lists; /* actual number of command lists */
    unsigned         next_cmd_list; /* round-robin index for load balancing */

    /* Pre-allocated event pool for performance optimization */
    ze_event_pool_handle_t
                   ze_event_pool; /* shared event pool for all operations */
    unsigned       event_pool_size; /* number of events in the pool */
    ucs_spinlock_t event_lock; /* lock for event allocation */
    uint64_t       *event_bitmap; /* bitmap to track free events */
} uct_ze_ipc_iface_t;


typedef struct uct_ze_ipc_event_desc {
    ze_event_handle_t      event;
    ze_event_pool_handle_t event_pool; /* NULL when using shared pool */
    void                   *mapped_addr;
    uct_completion_t       *comp;
    ucs_queue_elem_t       queue;
    int                    dup_fd; /* duplicated fd to close, or -1 if none */
    pid_t                  pid; /* remote process id for cache lookup */
    uintptr_t              address; /* remote base address for cache lookup */
    unsigned event_index; /* index in shared event pool, or -1 if using private pool */
} uct_ze_ipc_event_desc_t;


typedef struct uct_ze_ipc_flush_desc {
    uct_completion_t super; /* counts the flush markers still queued */
    uct_completion_t *comp; /* user completion of the flush */
} uct_ze_ipc_flush_desc_t;


/**
 * Allocate an event from the shared event pool, or from a private one-event
 * pool when the shared pool is exhausted
 *
 * @param iface      Pointer to ze_ipc interface
 * @param event_desc Descriptor to fill with the event, its pool and index
 * @return           UCS_OK on success, error status on Level Zero failure
 */
ucs_status_t uct_ze_ipc_alloc_event(uct_ze_ipc_iface_t *iface,
                                    uct_ze_ipc_event_desc_t *event_desc);

/**
 * Release an event allocated by uct_ze_ipc_alloc_event
 *
 * @param iface      Pointer to ze_ipc interface
 * @param event_desc Descriptor filled by uct_ze_ipc_alloc_event
 */
void uct_ze_ipc_free_event(uct_ze_ipc_iface_t *iface,
                           uct_ze_ipc_event_desc_t *event_desc);


#endif
