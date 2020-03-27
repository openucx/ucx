/**
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCS_MEMORY_TYPE_H_
#define UCS_MEMORY_TYPE_H_

#include <ucs/sys/compiler_def.h>
#include <ucs/sys/topo.h>

BEGIN_C_DECLS


/* Memory types accessible from CPU  */
#define UCS_MEMORY_TYPES_CPU_ACCESSIBLE \
    (UCS_BIT(UCS_MEMORY_TYPE_HOST) | \
     UCS_BIT(UCS_MEMORY_TYPE_CUDA_MANAGED) | \
     UCS_BIT(UCS_MEMORY_TYPE_ROCM_MANAGED))


/*
 * Memory types
 */
typedef enum ucs_memory_type {
    UCS_MEMORY_TYPE_HOST,          /**< Default system memory */
    UCS_MEMORY_TYPE_CUDA,          /**< NVIDIA CUDA memory */
    UCS_MEMORY_TYPE_CUDA_MANAGED,  /**< NVIDIA CUDA managed (or unified) memory*/
    UCS_MEMORY_TYPE_ROCM,          /**< AMD ROCM memory */
    UCS_MEMORY_TYPE_ROCM_MANAGED,  /**< AMD ROCM managed system memory */
    UCS_MEMORY_TYPE_LAST
} ucs_memory_type_t;


/**
 * Array of string names for each memory type
 */
extern const char *ucs_memory_type_names[];

/**
 * Array of string descriptions for each memory type
 */
extern const char *ucs_memory_type_descs[];

enum ucs_mem_info_field {
    UCS_MEM_INFO_MEM_TYPE = UCS_BIT(0),
    UCS_MEM_INFO_SYS_DEV  = UCS_BIT(1)
};

typedef struct ucs_mem_info {
    uint64_t          field_mask;
    ucs_memory_type_t mem_type;
    ucs_sys_device_t  sys_dev;
} ucs_mem_info_t;


END_C_DECLS

#endif
