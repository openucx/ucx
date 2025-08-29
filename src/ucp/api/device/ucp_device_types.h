/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_DEVICE_TYPES_H
#define UCP_DEVICE_TYPES_H

#include <uct/api/uct.h>


typedef struct ucp_mem_list_elem {
} ucp_device_mem_list_elem_t;


/**
 * @ingroup UCP_DEVICE
 * @brief Descriptor list handle stored on GPU memory.
 *
 * This handle is obtained and managed with functions called on host. It can be
 * used repeatedly from GPU code to perform memory transfers.
 *
 * The handle and most of its content is stored on GPU memory, with the intent
 * to be as memory-local as possible.
 */
typedef struct {
    /**
     * Allow runtime ABI compatibility checks, between host and device code.
     */
    int                        version;

    /**
     * Protocol index computed by host handle management functions when
     * creating handle.
     */
    int                        proto_idx;

    /**
     * Array of pointers to UCT exported endpoints, used for multi-lane
     * transfers.
     */
    uct_ep_h                   *uct_ep;

    /**
     * Number of UCT exported endpoints found in @a uct_ep array.
     */
    unsigned                   num_uct_eps;

    /**
     * Number of entries in the memory descriptors array @a elems.
     */
    unsigned                   mem_list_length;

    /**
     * Array of memory descriptors containing memory pairs to be used by device
     * functions for memory transfers.
     */
    ucp_device_mem_list_elem_t elems[];
} ucp_device_mem_list_handle_t;

typedef ucp_device_mem_list_handle_t *ucp_device_mem_list_handle_h;


/**
 * @ingroup UCP_DEVICE
 * @brief GPU request descriptor of a given batch
 *
 * This request tracks a batch of memory operations in progress. It can be used
 * with @ref ucp_device_progress_req to detect request completion.
 */
typedef struct ucp_device_request {
} ucp_device_request_t;

#endif /* UCP_DEVICE_TYPES_H */
