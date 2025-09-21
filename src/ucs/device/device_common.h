/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_DEVICE_COMMON_H
#define UCS_DEVICE_COMMON_H

#include <ucs/config/types.h>
#include <stdint.h>

#ifdef __NVCC__
#include "cuda_device.cuh"
#else
#include "stub_device.h"
#endif /* __NVCC__ */

BEGIN_C_DECLS

/* Logging configuration for device functions */
typedef struct {
    uint8_t level;
} ucs_device_log_config_t;


/**
 * @brief Cooperation level when calling device functions.
 */
typedef enum {
    UCS_DEVICE_LEVEL_THREAD = 0,
    UCS_DEVICE_LEVEL_WARP   = 1,
    UCS_DEVICE_LEVEL_BLOCK  = 2,
    UCS_DEVICE_LEVEL_GRID   = 3
} ucs_device_level_t;


/** Names for @ref ucs_device_level_t */
extern const char *ucs_device_level_names[];


/* Number of threads in a warp */
#define UCS_DEVICE_NUM_THREADS_IN_WARP 32


/* Initialize the logging configuration for device functions */
void ucs_device_log_config_init(ucs_device_log_config_t *config);

END_C_DECLS

#endif /* UCS_DEVICE_COMMON_H */
