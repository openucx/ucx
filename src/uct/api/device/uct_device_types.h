/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_DEVICE_TYPES_H
#define UCT_DEVICE_TYPES_H

#include <ucs/type/status.h>
#include <stdint.h>


/**
 * @defgroup UCT_DEVICE Device API
 * @ingroup UCT_API
 * * This section describes UCT Device API.
 * @{
 * @}
 */

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


/* Base structure for all device memory elements */
struct uct_device_mem_element {
};

#endif
