/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_CUDA_TYPES_H
#define UCP_CUDA_TYPES_H

#include <ucs/type/status.h>
#include <uct/api/uct.h>
#include <ucp/api/ucp.h>


/* TODO: Use from ucp.h */
typedef struct ucp_dlist_elem {
} ucp_dlist_elem_t;


/**
 * @ingroup UCP_COMM
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
    int              version;

    /**
     * Protocol index computed by host handle management functions when
     * creating handle.
     */
    int              proto_idx;

    /**
     * Array of pointers to UCT exported endpoints, used for multi-lane
     * transfers.
     */
    uct_ep_t         **uct_ep;

    /**
     * Number of UCT exported endpoints found in @ref uct_ep arrays.
     */
    unsigned         num_uct_eps;

    /**
     * Number of entries in the descriptor list array @ref elems.
     */
    unsigned         dlist_length;

    /**
     * Array of descriptor list containing memory pairs to be used by GPU
     * device functions for memory transfers.
     */
    ucp_dlist_elem_t elems[];
} ucp_dlist_handle_t;

#endif /* UCP_CUDA_TYPES_H */
