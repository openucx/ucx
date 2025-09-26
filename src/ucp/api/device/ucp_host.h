/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_HOST_H_
#define UCP_HOST_H_

#include <sys/types.h>
#include <ucs/sys/compiler_def.h>
#include <ucs/type/status.h>
#include <ucp/api/ucp_def.h>


BEGIN_C_DECLS

/**
 * @defgroup UCP_DEVICE Device API
 * @ingroup UCP_API
 * @{
 * UCP Device API.
 * @}
 */


/**
 * @ingroup UCP_DEVICE
 * @brief Memory descriptor list attributes field mask.
 *
 * The enumeration allows specifying which fields in @ref
 * ucp_device_mem_list_elem are present.
 *
 * It is used to enable backward compatibility support.
 */
enum ucp_device_mem_list_elem_field {
    UCP_DEVICE_MEM_LIST_ELEM_FIELD_MEMH        = UCS_BIT(0), /**< Source memory handle */
    UCP_DEVICE_MEM_LIST_ELEM_FIELD_RKEY        = UCS_BIT(1), /**< Unpacked remote memory key */
    UCP_DEVICE_MEM_LIST_ELEM_FIELD_LOCAL_ADDR  = UCS_BIT(2), /**< Local address */
    UCP_DEVICE_MEM_LIST_ELEM_FIELD_REMOTE_ADDR = UCS_BIT(3)  /**< Remote address */
};


/**
 * @ingroup UCP_DEVICE
 * @brief Memory descriptor list entry.
 *
 * This describes a pair of local and remote memory for which a memory operation
 * can later be performed multiple times, possibly with varying memory offsets.
 */
typedef struct ucp_device_mem_list_elem {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref ucp_device_mem_list_elem_field.
     * Fields not specified in this mask will be ignored.
     * Provides ABI compatibility with respect to adding new fields.
     */
    uint64_t   field_mask;

    /**
     * Local memory registration handle.
     */
    ucp_mem_h  memh;

    /**
     * Local memory address for the device transfer operations.
     */
    void*     local_addr;

    /**
     * Remote memory address for the device transfer operations.
     */
    uint64_t   remote_addr;

    /**
     * Unpacked memory key for the remote memory endpoint.
     */
    ucp_rkey_h rkey;
} ucp_device_mem_list_elem_t;


/**
 * @ingroup UCP_DEVICE
 * @brief Memory descriptor list create parameters field mask.
 *
 * The enumeration allows specifying which fields in @ref ucp_device_mem_list_params_t
 * are presents. It is used to enable backward compatibility support.
 */
enum ucp_device_mem_list_params_field {
    UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENTS     = UCS_BIT(0), /**< Elements array base address */
    UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENT_SIZE = UCS_BIT(1), /**< Element size in bytes */
    UCP_DEVICE_MEM_LIST_PARAMS_FIELD_NUM_ELEMENTS = UCS_BIT(2)  /**< Number of elements */
};


/**
 * @ingroup UCP_DEVICE
 * @brief Memory descriptor list create parameters.
 *
 * The structure defines the parameters that can be used to create a handle
 * with @ref ucp_device_mem_list_create.
 */
typedef struct ucp_device_mem_list_params {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref ucp_device_mem_list_params_field.
     * Fields not specified in this mask will be ignored.
     * Provides ABI compatibility with respect to adding new fields.
     */
    uint64_t                  field_mask;

    /**
     * Size in bytes of one descriptor element, for backward compatibility.
     */
    size_t                    element_size;

    /**
     * Number of elements presents in @a elements.
     */
    size_t                    num_elements;

    /**
     * Base address of the array of descriptor elements.
     */
    const ucp_device_mem_list_elem_t *elements;
} ucp_device_mem_list_params_t;


/**
 * @ingroup UCP_DEVICE
 * @brief Memory descriptor list create function for batched RMA operations.
 *
 * This function creates and populates a descriptor list handle using parameters
 * inputs from @ref ucp_device_mem_list_params_t. This descriptor is created for
 * the given remote endpoint. It can be used on a GPU using the corresponding
 * device functions.
 *
 * It can be used repeatedly, until finally released by calling @ref
 * ucp_device_mem_list_release.
 *
 * @param [in]  ep        Remote endpoint handle.
 * @param [in]  params    Parameters used to create the handle.
 * @param [out] handle    Created descriptor list handle.
 *
 * @return Error code as defined by @ref ucs_status_t.
 */
ucs_status_t
ucp_device_mem_list_create(ucp_ep_h ep,
                           const ucp_device_mem_list_params_t *params,
                           ucp_device_mem_list_handle_h *handle);


/**
 * @ingroup UCP_DEVICE
 * @brief Release function for a descriptor list handle.
 *
 * This function releases the handle that was created using @ref
 * ucp_device_mem_list_create.
 *
 * @param [in] handle     Created handle to release.
 */
void ucp_device_mem_list_release(ucp_device_mem_list_handle_h handle);


/**
 * @ingroup UCP_DEVICE
 * @brief Counter attributes field mask.
 *
 * The enumeration allows specifying which fields in @ref
 * ucp_device_counter_params_t are present. It is used to enable backward
 * compatibility support.
 */
enum ucp_device_counter_params_field {
    UCP_DEVICE_COUNTER_PARAMS_FIELD_MEM_TYPE = UCS_BIT(0), /**< Source memory handle */
    UCP_DEVICE_COUNTER_PARAMS_FIELD_MEMH     = UCS_BIT(1)  /**< Unpacked remote memory key */
};


/**
 * @ingroup UCP_DEVICE
 * @brief Parameters which can be used when calling @ref ucp_device_counter_init
 * and @ref ucp_device_counter_read.
 */
typedef struct ucp_device_counter_params {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref ucp_device_counter_init_params_field.
     * Fields not specified in this mask will be ignored.
     * Provides ABI compatibility with respect to adding new fields.
     */
    uint64_t          field_mask;

    /**
     * Optional memory type for the given @a counter memory area.
     */
    ucs_memory_type_t mem_type;

    /**
     * Optional memory registration handle for the given @a counter memory area.
     */
    ucp_mem_h         memh;
} ucp_device_counter_params_t;


/**
 * @ingroup UCP_DEVICE
 * @brief Initialize the contents of a counter memory area.
 *
 * This host routine is called by the receive side to set up the memory area for
 * signaling with a counter. A remote sender can then use the provided rkey to
 * notify when the data has been successfully sent.
 *
 * The receive side can poll for completion on this counter using the device
 * function @ref ucp_device_counter_read.
 *
 * The memory type or memory handle from params, might be used to help setting
 * the contents of the counting area.
 *
 * @param [in] worker      Worker to use when initializing a counter area.
 * @param [in] params      Parameters used to initialize the counter area.
 * @param [in] counter_ptr Address of the counting area.
 *
 * @return Error code as defined by @ref ucs_status_t.
 */
ucs_status_t ucp_device_counter_init(ucp_worker_h worker,
                                     const ucp_device_counter_params_t *params,
                                     void *counter_ptr);


/**
 * @ingroup UCP_DEVICE
 * @brief Read the value of a counter memory area.
 *
 * This host routine is called by the receive side to read the value of a counter
 * memory area.
 *
 * @param [in] worker      Worker to use when reading the counter value.
 * @param [in] params      Parameters used to read the counter value.
 * @param [in] counter_ptr Address of the counter memory area.
 *
 * @return Value of the counter memory area.
 */
uint64_t ucp_device_counter_read(ucp_worker_h worker,
                                 const ucp_device_counter_params_t *params,
                                 void *counter_ptr);

END_C_DECLS

#endif /* UCP_HOST_H */
