/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_DEVICE_TYPES_H
#define UCT_DEVICE_TYPES_H

#include <ucs/type/status.h>
#include <uct/api/uct_def.h>
#include <stdint.h>


/**
 * @defgroup UCT_DEVICE Device API
 * @ingroup UCT_API
 * * This section describes UCT Device API.
 * @{
 * @}
 */


/**
 * @brief Device memory element for CUDA IPC.
 */
typedef struct {
    ptrdiff_t mapped_offset;
} uct_cuda_ipc_device_mem_element_t;


/**
 * @brief Completion object for device CUDA IPC operations.
 */
typedef struct {
} uct_cuda_ipc_completion_t;


/**
 * @brief Device memory element for GDAKI.
 */
typedef struct uct_rc_gdaki_device_mem_element {
    uint32_t lkey;
    uint32_t rkey;
} uct_rc_gdaki_device_mem_element_t;


/**
 * @brief Specify modifier flags for device sending functions.
 */
typedef enum {
    UCT_DEVICE_FLAG_NODELAY = UCS_BIT(0) /**< Complete before return. */
} uct_device_flags_t;


/* Device transport id (for internal use) */
typedef enum {
    UCT_DEVICE_TL_RC_MLX5_GDA,
    UCT_DEVICE_TL_CUDA_IPC,
    UCT_DEVICE_TL_LAST
} uct_device_tl_id_t;


/* Base class for all device endpoints */
typedef struct uct_device_ep {
    uint8_t uct_tl_id; /* Defined in uct_device_tl_id_t */
} uct_device_ep_t;


/* Completion object for device operations */
typedef union uct_device_completion uct_device_completion_t;


/* Union of all uct device memory elements */
union uct_tl_device_mem_element {
    uct_rc_gdaki_device_mem_element_t gdaki_mem_element;
    uct_cuda_ipc_device_mem_element_t cuda_ipc_mem_element;
};

/* Base structure for all device memory elements */
struct uct_device_mem_element {
};

typedef struct uct_device_local_mem_list_elem {
    void                     *addr;
    uct_device_mem_element_t uct_mem_element;
} uct_device_local_mem_list_elem_t;

typedef struct uct_device_remote_mem_list_elem {
    uct_device_ep_h          device_ep;
    uint64_t                 addr;
    uct_device_mem_element_t uct_mem_element;
} uct_device_remote_mem_list_elem_t;

#endif
