/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCT_COMPONENT_H_
#define UCT_COMPONENT_H_

#include <uct/api/uct.h>
#include <ucs/config/parser.h>
#include <ucs/datastruct/list.h>


/* Forward declaration */
typedef struct uct_component uct_component_t;


/**
 * Keeps information about allocated configuration structure, to be used when
 * releasing the options.
 */
typedef struct uct_config_bundle {
    ucs_config_field_t *table;
    const char         *table_prefix;
    char               data[];
} uct_config_bundle_t;


/**
 * Component method to query component memory domain resources.
 *
 * @param [in]  component               Query memory domain resources for this
 *                                      component.
 * @param [out] resources_p             Filled with a pointer to an array of
 *                                      memory domain resources, which should be
 *                                      released with ucs_free().
 * @param [out] num_resources_p         Filled with the number of memory domain
 *                                      resource entries in the array.
 *
 * @return UCS_OK on success or error code in case of failure.
 */
typedef ucs_status_t (*uct_component_query_md_resources_func_t)(
                uct_component_t *component, uct_md_resource_desc_t **resources_p,
                unsigned *num_resources_p);


/**
 * Component method to open a memory domain.
 *
 * @param [in]  component               Open memory domain resources on this
 *                                      component.
 * @param [in]  md_name                 Name of the memory domain to open, as
 *                                      returned by
 *                                      @ref uct_component_query_resources_func_t
 * @param [in]  config                  Memory domain configuration.
 * @param [out] md_p                    Handle to the opened memory domain.
 *
 * @return UCS_OK on success or error code in case of failure.
 */
typedef ucs_status_t (*uct_component_md_open_func_t)(
                uct_component_t *component, const char *md_name,
                const uct_md_config_t *config, uct_md_h *md_p);


/**
 * Component method to open a client/server connection manager.
 *
 * @param [in]  component               Open a connection manager on this
 *                                      component.
 * @param [in]  worker                  Open the connection manager on this worker.
 * @param [in]  config                  Connection manager configuration.
 * @param [out] cm_p                    Filled with a handle to the connection manager.
 *
 * @return UCS_OK on success or error code in case of failure.
 */
typedef ucs_status_t (*uct_component_cm_open_func_t)(
                uct_component_t *component, uct_worker_h worker,
                const uct_cm_config_t *config, uct_cm_h *cm_p);


/**
 * Component method to unpack a remote key buffer into a remote key object.
 *
 * @param [in]  component               Unpack the remote key buffer on this
 *                                      component.
 * @param [in]  rkey_buffer             Remote key buffer to unpack.
 * @param [in]  config                  Memory domain configuration.
 * @param [out] rkey_p                  Filled with a pointer to the unpacked
 *                                      remote key.
 * @param [out] handle_p                Filled with an additional handle which
 *                                      is used to release the remote key, but
 *                                      is not required for remote memory
 *                                      access operations.
 *
 * @return UCS_OK on success or error code in case of failure.
 */
typedef ucs_status_t (*uct_component_rkey_unpack_func_t)(
                uct_component_t *component, const void *rkey_buffer,
                uct_rkey_t *rkey_p, void **handle_p);


/**
 * Component method to obtain a locally accessible pointer to a remote key.
 *
 * @param [in]  component               Get remote key memory pointer on this
 *                                      component.
 * @param [in]  rkey                    Obtain the pointer for this remote key.
 * @param [in]  handle                  Remote key handle, as returned from
 *                                      @ref uct_component_rkey_unpack_func_t.
 * @param [in]  remote_addr             Remote address to obtain the pointer for.
 * @param [out] local_addr_p            Filled with the local access pointer.
 *
 * @return UCS_OK on success or error code in case of failure.
 */
typedef ucs_status_t (*uct_component_rkey_ptr_func_t)(
                uct_component_t *component, uct_rkey_t rkey, void *handle,
                uint64_t remote_addr, void **local_addr_p);


/**
 * Component method to release an unpacked remote key.
 *
 * @param [in]  component               Release the remote key of this
 *                                      component.
 * @param [in]  rkey                    Release this remote key.
 * @param [in]  handle                  Remote key handle, as returned from
 *                                      @ref uct_component_rkey_unpack_func_t.
 *
 * @return UCS_OK on success or error code in case of failure.
 */
typedef ucs_status_t (*uct_component_rkey_release_func_t)(
                uct_component_t *component, uct_rkey_t rkey, void *handle);


/**
 * Component method to initialize VFS for memory domain.
 *
 * @param [in]  md                      Handle to the opened memory domain.
 */
typedef void (*uct_component_md_vfs_init_func_t)(uct_md_h md);


extern ucs_list_link_t uct_components_list;


/**
 * Defines a UCT component
 */
struct uct_component {
    const char                              name[UCT_COMPONENT_NAME_MAX]; /**< Component name */
    uct_component_query_md_resources_func_t query_md_resources; /**< Query memory domain resources method */
    uct_component_md_open_func_t            md_open;            /**< Memory domain open method */
    uct_component_cm_open_func_t            cm_open;            /**< Connection manager open method */
    uct_component_rkey_unpack_func_t        rkey_unpack;        /**< Remote key unpack method */
    uct_component_rkey_ptr_func_t           rkey_ptr;           /**< Remote key access pointer method */
    uct_component_rkey_release_func_t       rkey_release;       /**< Remote key release method */
    ucs_config_global_list_entry_t          md_config;          /**< MD configuration entry */
    ucs_config_global_list_entry_t          cm_config;          /**< CM configuration entry */
    ucs_list_link_t                         tl_list;            /**< List of transports */
    ucs_list_link_t                         list;               /**< Entry in global list of components */
    uint64_t                                flags;              /**< Flags as defined by
                                                                     UCT_COMPONENT_FLAG_xx */
    /**< Memory domain initialize VFS method */
    uct_component_md_vfs_init_func_t        md_vfs_init;
};


#define UCT_COMPONENT_NAME(_name) uct_##_name##_component


/**
 * Register a component for usage, so it will be returned from
 * @ref uct_query_components.
 *
 * @param [in] _component  Pointer to a global component structure to register.
 */
#define UCT_COMPONENT_REGISTER(_component) \
    UCS_STATIC_INIT { \
        ucs_list_add_tail(&uct_components_list, &(_component)->list); \
    } \
    UCS_CONFIG_REGISTER_TABLE_ENTRY(&(_component)->md_config, &ucs_config_global_list); \
    UCS_CONFIG_REGISTER_TABLE_ENTRY(&(_component)->cm_config, &ucs_config_global_list);


/**
 * Helper macro to initialize component's transport list head.
 */
#define UCT_COMPONENT_TL_LIST_INITIALIZER(_component) \
    UCS_LIST_INITIALIZER(&(_component)->tl_list, &(_component)->tl_list)


ucs_status_t uct_config_read(uct_config_bundle_t **bundle,
                             ucs_config_field_t *config_table,
                             size_t config_size, const char *env_prefix,
                             const char *cfg_prefix);

void uct_component_register(uct_component_t *component);

void uct_component_unregister(uct_component_t *component);

#endif
