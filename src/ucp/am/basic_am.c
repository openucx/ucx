/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <ucp/core/ucp_mm.h>
#include <ucp/core/ucp_ep.inl>
#include <uct/base/uct_iface.h>
#include <ucs/type/status.h>
#include <ucs/debug/log.h>
#include <inttypes.h>


// TODO: simplify interface
/**
 * @ingroup UCP_AM
 * @brief
 */
ucs_status_t ucp_worker_set_am_handler(ucp_worker_h worker, uct_am_callback_t cb, void *arg, uint32_t flags, uint8_t *id)
{
    ucs_status_t status = UCS_ERR_NO_RESOURCE;
    ucp_context_h context = worker->context;
    uct_base_iface_t *iface;
    uct_iface_attr_t *iface_attr;
    ucp_rsc_index_t tl_id;
    unsigned am_id;

    for (tl_id = 0; tl_id < context->num_tls; ++tl_id) {
        iface = ucs_derived_of(worker->ifaces[tl_id], uct_base_iface_t);
        iface_attr = &worker->iface_attrs[tl_id];
        if( !(iface_attr->cap.flags & UCT_IFACE_FLAG_AM_SHORT) ) continue;
        if( (flags & UCT_AM_CB_FLAG_SYNC) &&
          !(iface_attr->cap.flags & UCT_IFACE_FLAG_AM_CB_SYNC) ) continue;

        for (am_id = 0; am_id < UCT_AM_ID_MAX; ++am_id) {
            //TODO: better check for "default" am handlers
            if( iface->am[am_id].arg == (void*)(uintptr_t)am_id ) continue;
            status = uct_iface_set_am_handler(&iface->super, am_id,
                                              cb, arg,
                                              flags);
            if(UCS_OK != status) return status;
        }
    }
    return status;
}

/**
 * @ingroup UCP_AM
 * @brief
 */
//TODO: inline
ucs_status_t ucp_ep_am_short(ucp_ep_h ep, uint8_t id, uint64_t header,
                                            const void *payload, unsigned length)
{
    ucp_lane_index_t lane = ucp_ep_get_am_lane(ep);
    return uct_ep_am_short(ep->uct_eps[lane], id, header, payload, length);
}




#if 0
//TODO: decide simpler interface for long/ddt based am
/**
 * @ingroup UCT_AM
 * @brief
 */
UCT_INLINE_API ssize_t uct_ep_am_bcopy(uct_ep_h ep, uint8_t id,
uct_pack_callback_t pack_cb, void *arg)
{
return ep->iface->ops.ep_am_bcopy(ep, id, pack_cb, arg);
}


/**
* @ingroup UCT_AM
* @brief
*/
UCT_INLINE_API ucs_status_t uct_ep_am_zcopy(uct_ep_h ep, uint8_t id, void *header,
unsigned header_length, const void *payload,
size_t length, uct_mem_h memh,
uct_completion_t *comp)
{
return ep->iface->ops.ep_am_zcopy(ep, id, header, header_length, payload,
length, memh, comp);
}
# endif


