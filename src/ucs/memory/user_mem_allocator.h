/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (C) The University of Tennessee and The University 
*               of Tennessee Research Foundation. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_USER_MEM_ALLOCATOR_H
#define UCS_USER_MEM_ALLOCATOR_H

#include <ucs/type/status.h>

#include <stdint.h>
#include <stddef.h>

/**
  * @ingroup UCS_RESOURCE
 * @brief User Memory allocator initialization params.
 *
 * The structure defines the configuration of
 * the needed memory allocator.
 */
typedef struct ucs_user_mem_allocator_params {
    uint64_t field_mask;
    size_t seg_size;
    size_t data_offset;
} ucs_user_mem_allocator_params_t;


/**
* @ingroup UCS_RESOURCE
* @brief User defined memory allocation instance constructor. 
*
* @param [in]  params  Memory allocator configuration.
*
* @param [out]  arg    Opaque object representing memory allocation instance implemented by the user
*
* @return Error code as defined by @ref ucs_status_t
*/
typedef ucs_status_t (*ucs_user_mem_allocator_init_func_t)(const ucs_user_mem_allocator_params_t* params, void** arg);


/**
* @ingroup UCS_RESOURCE
* @brief Cleanup the User defined memory allocation instance.
*
* @return Error code as defined by @ref ucs_status_t
*/
typedef ucs_status_t (*ucs_user_mem_allocator_cleanup_func_t)(void* arg);


/**
* @ingroup UCS_RESOURCE
* @brief Free descriptor allocated using user memory allocator
*
* @param [in]   arg  Opaque object representing memory allocation instance implemented by the user
*
* @param [in]  desc  Descriptor to free.
*
* @return Error code as defined by @ref ucs_status_t
*/
typedef ucs_status_t (*ucs_user_mem_allocator_free_func_t)(void* arg, void *desc);


/**
* @ingroup UCS_RESOURCE
* @brief Get descriptor from user defined memory allocation instance 
*
* @param [in]   arg       Opaque object representing memory allocation instance implemented by the user
* 
* @param [out]  desc      Allocated descriptor.
*
* @param [out]  ucp_memh  UCP Memory handle 
*
* @return Error code as defined by @ref ucs_status_t
*/
typedef ucs_status_t (*ucs_user_mem_allocator_malloc_func_t)(void* arg, void **desc, void **ucp_memh);

#endif
