/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_H_
#define UCT_H_


#include "tl.h"


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
 * @brief Open a communication interface.
 *
 * @param [in]  context       Handle to context.
 * @param [in]  tl_name       Transport name.
 * @param [in]  dev_name      Hardware device name,
 * @param [out] iface_p       Filled with a handle to opened communication interface.
 *
 * @return Error code.
 */
ucs_status_t uct_iface_open(uct_context_h context, const char *tl_name,
                            const char *dev_name, uct_iface_h *iface_p);


/**
 * @ingroup CONTEXT
 *
 * @brief Unpack a remote key.
 *
 * @param [in]  context      Handle to context.
 * @param [in]  rkey_buffer  Packet remote key buffer.
 * @param [out] rkey_ob      Filled with the unpacked remote key and its type.
 *
 * @return Error code.
 */
ucs_status_t uct_rkey_unpack(uct_context_h context, void *rkey_buffer,
                             uct_rkey_bundle_t *rkey_ob);


/**
 * @ingroup CONTEXT
 *
 * @brief Unpack a remote key.
 *
 * @param [in]  context      Handle to context.
 * @param [in]  rkey_ob      Remote key to release.
 */
void uct_rkey_release(uct_context_h context, uct_rkey_bundle_t *rkey_ob);


static inline ucs_status_t uct_pd_query(uct_pd_h pd, uct_pd_attr_t *pd_attr)
{
    return pd->ops->query(pd, pd_attr);
}

static inline ucs_status_t uct_mem_map(uct_pd_h pd, void *address, size_t length,
                                       unsigned flags, uct_lkey_t *lkey_p)
{
    return pd->ops->mem_map(pd, address, length, flags, lkey_p);
}

static inline ucs_status_t uct_mem_unmap(uct_pd_h pd, uct_lkey_t lkey)
{
    return pd->ops->mem_unmap(pd, lkey);
}

static inline ucs_status_t uct_rkey_pack(uct_pd_h pd, uct_lkey_t lkey, void *rkey_buffer)
{
    return pd->ops->rkey_pack(pd, lkey, rkey_buffer);
}

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

#endif
