/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_DEVICE_TYPES_H
#define UCP_DEVICE_TYPES_H

#include <uct/api/device/uct_device_types.h>
#include <uct/api/uct_def.h>
#include <uct/api/uct.h>


#define UCP_DEVICE_MEM_LIST_MAX_EPS    2
#define UCP_DEVICE_MEM_LIST_VERSION_V1 1


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
typedef struct ucp_device_mem_list_handle {
    /**
     * Structure version. Allow runtime ABI compatibility checks between host
     * and device code.
     */
    uint16_t        version;

    /**
     * Protocol index computed by host handle management functions when
     * creating handle.
     */
    uint8_t         proto_idx;

    /**
     * Number of UCT device endpoints found in @a uct_ep array.
     */
    uint8_t         num_uct_eps;

    /**
     * Number of entries in the memory descriptors array @a elems.
     */
    uint32_t        mem_list_length;

    /**
     * Array of pointers to UCT device endpoints, used for multi-lane
     * transfers.
     */
    uct_device_ep_h uct_device_eps[UCP_DEVICE_MEM_LIST_MAX_EPS];

    /**
     * Size of a given UCT memory element object for each UCT.
     */
    uint16_t        uct_mem_element_size[UCP_DEVICE_MEM_LIST_MAX_EPS];

    /**
     * For each @ref num_uct_eps UCT endpoints, a list of @ref
     * uct_device_mem_element objects.
     */
} ucp_device_mem_list_handle_t;

#endif /* UCP_DEVICE_TYPES_H */
