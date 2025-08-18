/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef GDAKI_MEM_ACCESS_H_
#define GDAKI_MEM_ACCESS_H_

#include <stddef.h>
#include <stdbool.h>
#include <gdrapi.h>
#include <ucs/sys/compiler_def.h>

BEGIN_C_DECLS

// Opaque handle for memory access
typedef struct gdaki_mem_handle *gdaki_mem_handle_t;

/**
 * Create a GDRcopy memory access handle for GPU memory
 * If GDRcopy is not available, it will use managed memory.
 *
 * @param [in]  gpu_ptr     Pointer to GPU memory, or NULL to allocate new memory
 * @param [in]  size        Size of the memory region
 *
 * @return Handle for memory access, or NULL if failed
 *         If gpu_ptr is NULL, new GPU memory will be allocated and initialized to zero
 */
gdaki_mem_handle_t gdaki_mem_create(void *gpu_ptr, size_t size);

/**
 * Get CPU accessible pointer through GDRcopy mapping
 *
 * @param [in]  handle      Memory access handle
 *
 * @return Direct mapped memory pointer accessible from CPU, or NULL if handle is invalid
 */
void *gdaki_mem_get_ptr(gdaki_mem_handle_t handle);

/**
 * Get the GPU memory pointer
 *
 * @param [in]  handle      Memory access handle
 *
 * @return GPU memory pointer, or NULL if handle is invalid
 */
void *gdaki_mem_get_gpu_ptr(gdaki_mem_handle_t handle);

/**
 * Release memory handle and associated resources
 * If GPU memory was allocated by gdaki_mem_create, it will be freed
 *
 * @param [in]  handle      Memory access handle
 */
void gdaki_mem_destroy(gdaki_mem_handle_t handle);

END_C_DECLS

#endif /* GDAKI_MEM_ACCESS_H_ */
