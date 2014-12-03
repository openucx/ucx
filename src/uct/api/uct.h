/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_H_
#define UCT_H_

#include <uct/api/tl.h>
#include <uct/api/version.h>
#include <ucs/config/types.h>
#include <stdio.h>


/**
 * @ingroup CONTEXT
 * @brief Initialize global context.
 *
 * @param [out] context_p   Filled with context handle.
 *
 * @return Error code.
 */
ucs_status_t uct_init(uct_context_h *context_p);


/**
 * @ingroup CONTEXT
 * @brief Destroy global context.
 *
 * @param [in] context   Handle to context.
 */
void uct_cleanup(uct_context_h context);


/**
 * @ingroup CONTEXT
 * @brief Progress all communications of the context.
 *
 * @param [in] context   Handle to context.
 */
void uct_progress(uct_context_h context);


/**
 * @ingroup CONTEXT
 * @brief Query for transport resources.
 *
 * @param [in]  context         Handle to context.
 * @param [out] resources_p     Filled with a pointer to an array of resource descriptors.
 * @param [out] num_resources_p Filled with the number of resources in the array.
 *
 * @return Error code.
 */
ucs_status_t uct_query_resources(uct_context_h context,
                                 uct_resource_desc_t **resources_p,
                                 unsigned *num_resources_p);


/**
 * @ingroup CONTEXT
 * @brief Release the list of resources returned from uct_query_resources.
 *
 * @param [in] resources  Array of resource descriptors to release.
 *
 */
void uct_release_resource_list(uct_resource_desc_t *resources);


/**
 * @ingroup CONTEXT
 * @brief Read transport-specific interface configuration.
 *
 * @param [in]  context       Handle to context.
 * @param [in]  tl_name       Transport name.
 * @param [in]  env_prefix    If non-NULL, search for environment variables
 *                            starting with this UCT_<prefix>_. Otherwise, search
 *                            for environment variables starting with just UCT_.
 * @param [in]  filename      If non-NULL, read configuration from this file. If
 *                            the file does not exist, it will be ignored.
 * @param [out] config_p      Filled with a pointer to configuration.
 *
 * @return Error code.
 */
ucs_status_t uct_iface_config_read(uct_context_h context, const char *tl_name,
                                   const char *env_prefix, const char *filename,
                                   uct_iface_config_t **config_p);


/**
 * @ingroup CONTEXT
 * @brief Release configuration memory returned from uct_iface_read_config().
 *
 * @param [in]  config        Configuration to release.
 */
void uct_iface_config_release(uct_iface_config_t *config);


/**
 * @ingroup CONTEXT
 * @brief Print interface configuration to a stream.
 *
 * @param [in]  config        Configuration to print.
 * @param [in]  stream        Output stream to print to.
 * @param [in]  title         Title to the output.
 * @param [in]  print_flags   Controls how the configuration is printed.
 */
void uct_iface_config_print(uct_iface_config_t *config, FILE *stream,
                            const char *title, ucs_config_print_flags_t print_flags);


/**
 * @ingroup CONTEXT
 * @brief Print interface configuration to a stream.
 *
 * @param [in]  config        Configuration to release.
 * @param [in]  name          Configuration variable name.
 * @param [in]  value         Value to set.
 *
 * @return Error code.
 */
ucs_status_t uct_iface_config_modify(uct_iface_config_t *config,
                                     const char *name, const char *value);


/**
 * @ingroup CONTEXT
 * @brief Open a communication interface.
 *
 * @param [in]  context       Handle to context.
 * @param [in]  tl_name       Transport name.
 * @param [in]  dev_name      Hardware device name,
 * @param [in]  rx_headroom   How much bytes to reserve before the receive segment.
 * @param [in]  config        Interface configuration options. Should be obtained
 *                            from uct_iface_read_config() function, or point to
 *                            transport-specific structure which extends uct_iface_config_t.
 * @param [out] iface_p       Filled with a handle to opened communication interface.
 *
 * @return Error code.
 */
ucs_status_t uct_iface_open(uct_context_h context, const char *tl_name,
                            const char *dev_name, size_t rx_headroom,
                            uct_iface_config_t *config, uct_iface_h *iface_p);


/**
 * @ingroup CONTEXT
 * @brief Set active message handler for the interface.
 *
 * @param [in]  iface    Interface to set the active message handler for.
 * @param [in]  id       Active message id. Must be 0..UCT_AM_ID_MAX-1.
 * @param [in]  cb       Active message callback. NULL to clear.
 * @param [in]  arg      Active message argument.
 */
ucs_status_t uct_set_am_handler(uct_iface_h iface, uint8_t id,
                                uct_am_callback_t cb, void *arg);


/**
 * @ingroup CONTEXT
 * @brief Query for protection domain attributes..
 *
 * @param [in]  pd       Protection domain to query.
 * @param [out] pd_attr  Filled with protection domain attributes.
 */
ucs_status_t uct_pd_query(uct_pd_h pd, uct_pd_attr_t *pd_attr);


/**
 * @ingroup CONTEXT
 * @brief Map memory for zero-copy sends and remote access.
 *
 * @param [in]  pd       Protection domain to map memory on.
 * @param [in]  address  Address to map.
 * @param [in]  length   Range length to map.
 * @param [in]  flags    Mapping flags (currently reserved - set to 0).
 * @param [out] lkey_p   Filled with local access key for mapped region.
 */
ucs_status_t uct_mem_map(uct_pd_h pd, void *address, size_t length,
                         unsigned flags, uct_lkey_t *lkey_p);


/**
 * @ingroup CONTEXT
 * @brief Allocate memory which can be used for for zero-copy sends and remote access.
 *
 * @param [in]     pd         Protection domain to map memory on.
 * @param [inout]  length_p   How many bytes to allocate. Filled with the actual
 *                             allocated size, which is larger than or equal to the
 *                             requested size.
 * @param [in]     flags      Allocation flags (currently reserved - set to 0).
 * @param [out]    address_p  Filled with a pointer to allocated region.
 * @param [out]    lkey_p     Filled with local access key for allocated region.
 */
ucs_status_t uct_mem_alloc(uct_pd_h pd, size_t *length_p, unsigned flags,
                           void **address_p, uct_lkey_t *lkey_p);


/**
 * @ingroup CONTEXT
 * @brief Either unmap memory passed to uct_mem_map, or release memory returned
 *        from uct_mem_alloc.
 *
 * @param [in]  pd          Protection domain which was used to allocate/map the memory.
 * @paran [in]  lkey        Local access key to memory region.
 */
ucs_status_t uct_mem_unmap(uct_pd_h pd, uct_lkey_t lkey);


/**
 * @ingroup CONTEXT
 *
 * @brief Pack a remote key.
 *
 * @param [in]  pd           Handle to protection domain.
 * @param [in]  lkey         Local key, whose remote key should be packed.
 * @param [out] rkey_buffer  Filled with packed remote key.
 *
 * @return Error code.
 */
ucs_status_t uct_rkey_pack(uct_pd_h pd, uct_lkey_t lkey, void *rkey_buffer);


/**
 * @ingroup CONTEXT
 *
 * @brief Unpack a remote key.
 *
 * @param [in]  context      Handle to context.
 * @param [in]  rkey_buffer  Packed remote key buffer.
 * @param [out] rkey_ob      Filled with the unpacked remote key and its type.
 *
 * @return Error code.
 */
ucs_status_t uct_rkey_unpack(uct_context_h context, void *rkey_buffer,
                             uct_rkey_bundle_t *rkey_ob);


/**
 * @ingroup CONTEXT
 *
 * @brief Release a remote key.
 *
 * @param [in]  context      Handle to context.
 * @param [in]  rkey_ob      Remote key to release.
 */
void uct_rkey_release(uct_context_h context, uct_rkey_bundle_t *rkey_ob);


static inline ucs_status_t uct_iface_query(uct_iface_h iface,
                                           uct_iface_attr_t *iface_attr)
{
    return iface->ops.iface_query(iface, iface_attr);
}

static inline ucs_status_t uct_iface_get_address(uct_iface_h iface,
                                                 uct_iface_addr_t *iface_addr)
{
    return iface->ops.iface_get_address(iface, iface_addr);
}

static inline ucs_status_t uct_iface_flush(uct_iface_h iface, uct_req_h *req_p,
                                           uct_completion_cb_t cb)
{
    return iface->ops.iface_flush(iface, req_p, cb);
}

static inline void uct_iface_close(uct_iface_h iface)
{
    iface->ops.iface_close(iface);
}

static inline ucs_status_t uct_ep_create(uct_iface_h iface, uct_ep_h *ep_p)
{
    return iface->ops.ep_create(iface, ep_p);
}

static inline void uct_ep_destroy(uct_ep_h ep)
{
    ep->iface->ops.ep_destroy(ep);
}

static inline ucs_status_t uct_ep_get_address(uct_ep_h ep, uct_ep_addr_t *ep_addr)
{
    return ep->iface->ops.ep_get_address(ep, ep_addr);
}

static inline ucs_status_t uct_ep_connect_to_iface(uct_ep_h ep, uct_iface_addr_t *iface_addr)
{
    return ep->iface->ops.ep_connect_to_iface(ep, iface_addr);
}

static inline ucs_status_t uct_ep_connect_to_ep(uct_ep_h ep, uct_iface_addr_t *iface_addr,
                                                uct_ep_addr_t *ep_addr)
{
    return ep->iface->ops.ep_connect_to_ep(ep, iface_addr, ep_addr);
}

static inline ucs_status_t uct_ep_put_short(uct_ep_h ep, void *buffer, unsigned length,
                                            uint64_t remote_addr, uct_rkey_t rkey)
{
    return ep->iface->ops.ep_put_short(ep, buffer, length, remote_addr, rkey);
}

static inline ucs_status_t uct_ep_am_short(uct_ep_h ep, uint8_t id, uint64_t header,
                                           void *payload, unsigned length)
{
    return ep->iface->ops.ep_am_short(ep, id, header, payload, length);
}

#endif
