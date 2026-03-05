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


#define UCP_DEVICE_MEM_LIST_VERSION_V1 1


/**
 * @ingroup UCP_DEVICE
 * @brief Descriptor list handle for remote memory stored on GPU memory.
 *
 * This handle is obtained and managed with functions called on host. It can be
 * used repeatedly from GPU code to perform memory transfers.
 *
 * The handle and most of its content is stored on GPU memory, with the intent
 * to be as memory-local as possible.
 */
typedef struct ucp_device_remote_mem_list {
    /**
     * Structure version. Allow runtime ABI compatibility checks between host
     * and device code.
     */
    uint16_t                          version;

    /**
     * Number of entries in the memory descriptors array @a elems.
     */
    uint32_t                          length;

    /**
     * UCT memory element objects are allocated contiguously.
     */
    uct_device_remote_mem_list_elem_t mem_elements[0];
} ucp_device_remote_mem_list_t;


/**
 * @ingroup UCP_DEVICE
 * @brief Descriptor list handle for local memory stored on GPU memory.
 *
 * This handle is obtained and managed with functions called on host. It can be
 * used repeatedly from GPU code to perform memory transfers.
 *
 * The handle and most of its content is stored on GPU memory, with the intent
 * to be as memory-local as possible.
 */
typedef struct ucp_device_local_mem_list {
    /**
     * Structure version. Allow runtime ABI compatibility checks between host
     * and device code.
     */
    uint16_t                         version;

    /**
     * Number of entries in the memory descriptors array @a elems.
     */
    uint32_t                         length;

    uct_device_local_mem_list_elem_t mem_elements[0];
} ucp_device_local_mem_list_t;

#endif /* UCP_DEVICE_TYPES_H */
