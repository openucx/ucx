/**
 * Copyright (C) Intel Corporation, 2023-2024. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCM_ZEMEM_H_
#define UCM_ZEMEM_H_

#include <level_zero/ze_api.h>


ze_result_t ucm_zeMemAllocHost(ze_context_handle_t context,
                               const ze_host_mem_alloc_desc_t *host_desc,
                               size_t size, size_t alignment, void **pptr);


ze_result_t ucm_zeMemAllocDevice(ze_context_handle_t context,
                                 const ze_device_mem_alloc_desc_t *device_desc,
                                 size_t size, size_t alignment,
                                 ze_device_handle_t device, void **pptr);


ze_result_t ucm_zeMemAllocShared(ze_context_handle_t context,
                                 const ze_device_mem_alloc_desc_t *device_desc,
                                 const ze_host_mem_alloc_desc_t *host_desc,
                                 size_t size, size_t alignment,
                                 ze_device_handle_t device, void **pptr);


ze_result_t ucm_zeMemFree(ze_context_handle_t context, void *ptr);


#endif
