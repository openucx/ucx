/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_DEVICE_DEF_H_
#define UCP_DEVICE_DEF_H_


/**
 * @ingroup UCP_COMM
 * @brief Opaque descriptor list handle from @ref ucp_gpu_mem_list_create.
 *
 * This handle is opaque from host point of view. It is to be used from a GPU
 * kernel using device specific functions.
 */
struct ucp_gpu_mem_list_handle;
typedef struct ucp_gpu_mem_list_handle *ucp_gpu_mem_list_handle_h;

#endif /* UCP_DEVICE_DEF_H_ */
