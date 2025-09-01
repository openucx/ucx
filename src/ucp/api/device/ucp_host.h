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
 * The enumeration allows specifying which fields in @ref ucp_mem_list_elem are
 * present. It is used to enable backward compatibility support.
 */
enum ucp_mem_list_elem_field {
    UCP_MEM_LIST_ELEM_FIELD_MEMH = UCS_BIT(0), /**< Source memory handle */
    UCP_MEM_LIST_ELEM_FIELD_RKEY = UCS_BIT(1)  /**< Unpacked remote memory key */
};


/**
 * @ingroup UCP_DEVICE
 * @brief Memory descriptor list entry.
 *
 * This describes a pair of local and remote memory for which a memory operation
 * can later be performed multiple times, possibly with varying memory offsets.
 */
typedef struct ucp_mem_list_elem {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref ucp_mem_list_elem_field.
     * Fields not specified in this mask will be ignored.
     * Provides ABI compatibility with respect to adding new fields.
     */
    uint64_t   field_mask;

    /**
     * Local memory registration handle.
     */
    ucp_mem_h  memh;

    /**
     * Unpacked memory key for the remote memory endpoint.
     */
    ucp_rkey_h rkey;
} ucp_mem_list_elem_t;


/**
 * @ingroup UCP_DEVICE
 * @brief Memory descriptor list create parameters field mask.
 *
 * The enumeration allows specifying which fields in @ref ucp_mem_list_params_t
 * are presents. It is used to enable backward compatibility support.
 */
enum ucp_mem_list_params_field {
    UCP_MEM_LIST_PARAMS_FIELD_ELEMENTS     = UCS_BIT(0), /**< Elements array base address */
    UCP_MEM_LIST_PARAMS_FIELD_ELEMENT_SIZE = UCS_BIT(1), /**< Element size in bytes */
    UCP_MEM_LIST_PARAMS_FIELD_NUM_ELEMENTS = UCS_BIT(2)  /**< Number of elements */
};


/**
 * @ingroup UCP_DEVICE
 * @brief Memory descriptor list create parameters.
 *
 * The structure defines the parameters that can be used to create a handle
 * with @ref ucp_mem_list_create.
 */
typedef struct ucp_mem_list_params {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref ucp_mem_list_params_field.
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
    const ucp_mem_list_elem_t *elements;
} ucp_mem_list_params_t;


/**
 * @ingroup UCP_DEVICE
 * @brief Opaque descriptor list stored on GPU.
 *
 * Host side does not have access to the content of this descriptor.
 */
struct ucp_device_mem_list_handle;
typedef struct ucp_device_mem_list_handle *ucp_device_mem_list_handle_h;


/**
 * @ingroup UCP_DEVICE
 * @brief Memory descriptor list create function for batched RMA operations.
 *
 * This function creates and populates a descriptor list handle using parameters
 * inputs from @ref ucp_mem_list_params_t. This descriptor is created for
 * the given remote endpoint. It can be used on a GPU using the corresponding
 * device functions.
 *
 * It can be used repeatedly, until finally released by calling @ref
 * ucp_mem_list_release.
 *
 * @param [in]  ep        Remote endpoint handle.
 * @param [in]  params    Parameters used to create the handle.
 * @param [out] handle    Created descriptor list handle.
 *
 * @return Error code as defined by @ref ucs_status_t.
 */
ucs_status_t
ucp_mem_list_create(ucp_ep_h ep,
                    const ucp_mem_list_params_t *params,
                    ucp_device_mem_list_handle_h *handle);


/**
 * @ingroup UCP_DEVICE
 * @brief Release function for a descriptor list handle.
 *
 * This function releases the handle that was created using @ref
 * ucp_mem_list_create.
 *
 * @param [in] handle     Created handle to release.
 */
void ucp_mem_list_release(ucp_device_mem_list_handle_h handle);


/**
 * @ingroup UCP_DEVICE
 * @brief UCP signal attributes field mask.
 *
 * This allows specifying which fields in @ref ucp_signal_attr_t are queried.
 */
enum ucp_signal_attr_field {
    /* The @ref ucp_signal_attr_t::signal_size field is queried. */
    UCP_SIGNAL_ATTR_FIELD_SIGNAL_SIZE = UCS_BIT(0)
};

/**
 * @ingroup UCP_DEVICE
 * @brief UCP signal attributes.
 *
 * The structure defines the attributes for the signaling functionality of the
 * UCP device API.
 */
typedef struct ucp_signal_attr {
    /**
     * Mask of valid fields in this structure, using bits from @ref
     * ucp_signal_attr_field. Fields not specified in this mask will be
     * ignored. Provides ABI compatibility with respect to adding new fields.
     *
     * The caller must set this field to indicate which attributes
     * they want to query. Only the requested fields will be populated
     * in the structure.
     */
    uint64_t field_mask;

    /**
     * Size of the signal structure used by UCP device API for signaling
     * operations.
     *
     * This field is an output parameter that indicates the size required
     * by a signaling memory area. It can be used for signal area allocation,
     * which can in turn be initialized with @ref ucp_signal_init.
     */
    size_t   signal_size;
} ucp_signal_attr_t;


/**
 * @ingroup UCP_DEVICE
 * @brief Signal query parameters structure.
 *
 * The structure defines the optional parameters that can be used to query the
 * signaling memory area properties with @ref ucp_signal_query.
 */
typedef struct ucp_signal_query_params {
    /**
     * Mask of valid fields in this structure. Fields not specified in this
     * mask will be ignored. Provides ABI compatibility with respect to adding
     * new fields.
     */
    uint64_t field_mask;
} ucp_signal_query_params_t;


/**
 * @ingroup UCP_DEVICE
 * @brief Query signal attributes.
 *
 * This host routine of the device API fetches information about signaling
 * memory area attributes. Those attributes can later be used for resource
 * allocation and configuration.
 *
 * @param [in]     context  Context to use.
 * @param [in]     params   Optional parameters to use for query.
 * @param [in/out] attr     Filled with signal attributes.
 *
 * @return Error code as defined by @ref ucs_status_t.
 */
ucs_status_t ucp_signal_query(ucp_context_h context_p,
                              const ucp_signal_query_params_t *params,
                              ucp_signal_attr_t *attr);


/**
 * @ingroup UCP_DEVICE
 * @brief Signal init attributes field mask.
 *
 * The enumeration allows specifying which fields in @ref
 * ucp_signal_init_params_t are present. It is used to enable backward
 * compatibility support.
 */
enum ucp_signal_init_params_field {
    UCP_SIGNAL_INIT_PARAMS_FIELD_MEM_TYPE = UCS_BIT(0), /**< Source memory handle */
    UCP_SIGNAL_INIT_PARAMS_FIELD_MEMH     = UCS_BIT(1)  /**< Unpacked remote memory key */
};


/**
 * @ingroup UCP_DEVICE
 * @brief Parameters which can be used when calling @ref ucp_signal_init.
 */
typedef struct ucp_signal_init_params {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref ucp_signal_init_params_field.
     * Fields not specified in this mask will be ignored.
     * Provides ABI compatibility with respect to adding new fields.
     */
    uint64_t          field_mask;

    /**
     * Optional memory type for the given @a signal area.
     */
    ucs_memory_type_t mem_type;

    /**
     * Optional memory registration handle for the given @a signal area.
     */
    ucp_mem_h         memh;
} ucp_signal_init_params_t;


/**
 * @ingroup UCP_DEVICE
 * @brief Initialize the content of a signaling memory area.
 *
 * This routine is called by the receive side to set up the memory area for
 * signaling. A remote sender can then use the provided rkey to notify when the
 * data has been successfully sent.
 *
 * The receive side can poll for completion on this signal using the device
 * function @ref ucp_device_counter_read.
 *
 * The memory type or memory handle from params, might be used to help setting
 * the content of the signal area.
 *
 * @param [in] context   Context to use when initializing a signaling area.
 * @param [in] params    Parameters used to initialize the signal.
 * @param [in] signal    Address of signaling area.
 *
 * @return Error code as defined by @ref ucs_status_t.
 */
ucs_status_t ucp_signal_init(ucp_context_t *context,
                             const ucp_signal_init_params_t *params,
                             void *signal);

END_C_DECLS

#endif /* UCP_HOST_H */
