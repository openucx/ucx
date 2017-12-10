/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucp_proxy_ep.h"
#include "ucp_ep.inl"

#include <ucs/debug/log.h>


#define UCP_PROXY_EP_PASTE_ARG_NAME(_, _index) \
    , UCS_PP_TOKENPASTE(arg, _index)

#define UCP_PROXY_EP_PASTE_ARG_TYPE(_, _bundle) \
    , UCS_PP_TUPLE_1 _bundle UCS_PP_TOKENPASTE(arg, UCS_PP_TUPLE_0 _bundle)

/* Generate list of typed arguments for a proxy function prototype */
#define UCP_PROXY_EP_FUNC_ARGS(...) \
    uct_ep_h ep \
    UCS_PP_FOREACH(UCP_PROXY_EP_PASTE_ARG_TYPE, _, \
                   UCS_PP_ZIP((UCS_PP_SEQ(UCS_PP_NUM_ARGS(__VA_ARGS__))), \
                              (__VA_ARGS__)))

/* Generate a list of arguments passed to function call */
#define UCP_PROXY_EP_FUNC_CALL(...) \
    proxy_ep->uct_ep \
    UCS_PP_FOREACH(UCP_PROXY_EP_PASTE_ARG_NAME, _, \
                   UCS_PP_SEQ(UCS_PP_NUM_ARGS(__VA_ARGS__)))

/* Generate a return statement based on return type */
#define UCP_PROXY_EP_RETURN(_retval) \
    UCS_PP_TOKENPASTE(UCP_PROXY_EP_RETURN_, _retval)

#define UCP_PROXY_EP_RETURN_ucs_status_t     return
#define UCP_PROXY_EP_RETURN_ucs_status_ptr_t return
#define UCP_PROXY_EP_RETURN_ssize_t          return
#define UCP_PROXY_EP_RETURN_void


/*
 * Define a proxy endpoint operation, which redirects the call to the underlying
 * transport endpoint.
 */
#define UCP_PROXY_EP_DEFINE_OP(_retval, _name, ...) \
    static _retval ucp_proxy_ep_##_name(UCP_PROXY_EP_FUNC_ARGS(__VA_ARGS__)) \
    { \
        ucp_proxy_ep_t *proxy_ep = ucs_derived_of(ep, ucp_proxy_ep_t); \
        UCP_PROXY_EP_RETURN(_retval) \
            uct_ep_##_name(UCP_PROXY_EP_FUNC_CALL(__VA_ARGS__)); \
    }


UCP_PROXY_EP_DEFINE_OP(ucs_status_t, put_short, const void*, unsigned,
                       uint64_t, uct_rkey_t)
UCP_PROXY_EP_DEFINE_OP(ssize_t, put_bcopy, uct_pack_callback_t, void*,
                       uint64_t, uct_rkey_t)
UCP_PROXY_EP_DEFINE_OP(ucs_status_t, put_zcopy, const uct_iov_t*, size_t,
                       uint64_t, uct_rkey_t, uct_completion_t*)
UCP_PROXY_EP_DEFINE_OP(ucs_status_t, get_bcopy, uct_unpack_callback_t, void*,
                       size_t, uint64_t, uct_rkey_t, uct_completion_t*)
UCP_PROXY_EP_DEFINE_OP(ucs_status_t, get_zcopy, const uct_iov_t*, size_t,
                       uint64_t, uct_rkey_t, uct_completion_t*)
UCP_PROXY_EP_DEFINE_OP(ucs_status_t, am_short, uint8_t, uint64_t, const void*,
                       unsigned)
UCP_PROXY_EP_DEFINE_OP(ssize_t, am_bcopy, uint8_t, uct_pack_callback_t, void*,
                       unsigned)
UCP_PROXY_EP_DEFINE_OP(ucs_status_t, am_zcopy, uint8_t, const void*, unsigned,
                       const uct_iov_t*, size_t, unsigned, uct_completion_t*)
UCP_PROXY_EP_DEFINE_OP(ucs_status_t, atomic_add64, uint64_t, uint64_t,
                       uct_rkey_t)
UCP_PROXY_EP_DEFINE_OP(ucs_status_t, atomic_fadd64, uint64_t, uint64_t,
                       uct_rkey_t, uint64_t*, uct_completion_t*)
UCP_PROXY_EP_DEFINE_OP(ucs_status_t, atomic_swap64, uint64_t, uint64_t,
                       uct_rkey_t, uint64_t*, uct_completion_t*)
UCP_PROXY_EP_DEFINE_OP(ucs_status_t, atomic_cswap64, uint64_t, uint64_t,
                       uint64_t, uct_rkey_t, uint64_t*, uct_completion_t*)
UCP_PROXY_EP_DEFINE_OP(ucs_status_t, atomic_add32, uint32_t, uint64_t,
                       uct_rkey_t)
UCP_PROXY_EP_DEFINE_OP(ucs_status_t, atomic_fadd32, uint32_t, uint64_t,
                       uct_rkey_t, uint32_t*, uct_completion_t*)
UCP_PROXY_EP_DEFINE_OP(ucs_status_t, atomic_swap32, uint32_t, uint64_t,
                       uct_rkey_t, uint32_t*, uct_completion_t*)
UCP_PROXY_EP_DEFINE_OP(ucs_status_t, atomic_cswap32, uint32_t, uint32_t,
                       uint64_t, uct_rkey_t, uint32_t*, uct_completion_t*)
UCP_PROXY_EP_DEFINE_OP(ucs_status_t, tag_eager_short, uct_tag_t, const void*,
                       size_t)
UCP_PROXY_EP_DEFINE_OP(ssize_t, tag_eager_bcopy, uct_tag_t, uint64_t,
                       uct_pack_callback_t, void*, unsigned)
UCP_PROXY_EP_DEFINE_OP(ucs_status_t, tag_eager_zcopy, uct_tag_t, uint64_t,
                       const uct_iov_t*, size_t, unsigned, uct_completion_t*)
UCP_PROXY_EP_DEFINE_OP(ucs_status_ptr_t, tag_rndv_zcopy, uct_tag_t, const void*,
                       unsigned, const uct_iov_t*, size_t, unsigned,
                       uct_completion_t*)
UCP_PROXY_EP_DEFINE_OP(ucs_status_t, tag_rndv_cancel, void*)
UCP_PROXY_EP_DEFINE_OP(ucs_status_t, tag_rndv_request, uct_tag_t, const void*,
                       unsigned, unsigned)
UCP_PROXY_EP_DEFINE_OP(ucs_status_t, pending_add, uct_pending_req_t*)
UCP_PROXY_EP_DEFINE_OP(void, pending_purge, uct_pending_purge_callback_t, void*)
UCP_PROXY_EP_DEFINE_OP(ucs_status_t, flush, unsigned, uct_completion_t*)
UCP_PROXY_EP_DEFINE_OP(ucs_status_t, fence, unsigned)
UCP_PROXY_EP_DEFINE_OP(ucs_status_t, check, unsigned, uct_completion_t*)
UCP_PROXY_EP_DEFINE_OP(ucs_status_t, get_address, uct_ep_addr_t*)
UCP_PROXY_EP_DEFINE_OP(ucs_status_t, connect_to_ep, const uct_device_addr_t*,
                       const uct_ep_addr_t*)
static UCS_CLASS_DEFINE_NAMED_DELETE_FUNC(ucp_proxy_ep_destroy, ucp_proxy_ep_t,
                                          uct_ep_t);

static ucs_status_t ucp_proxy_ep_fatal(uct_iface_h iface, ...)
{
    ucs_bug("unsupported function on proxy endpoint");
    return UCS_ERR_UNSUPPORTED;
}

UCS_CLASS_INIT_FUNC(ucp_proxy_ep_t, const uct_iface_ops_t *ops, ucp_ep_h ucp_ep,
                    uct_ep_h uct_ep, int is_owner)
{
    #define UCP_PROXY_EP_SET_OP(_name) \
        self->iface.ops._name = (ops->_name != NULL) ? ops->_name : ucp_proxy_##_name

    self->super.iface = &self->iface;
    self->ucp_ep      = ucp_ep;
    self->uct_ep      = uct_ep;
    self->is_owner    = is_owner;

    UCP_PROXY_EP_SET_OP(ep_put_short);
    UCP_PROXY_EP_SET_OP(ep_put_short);
    UCP_PROXY_EP_SET_OP(ep_put_bcopy);
    UCP_PROXY_EP_SET_OP(ep_put_zcopy);
    UCP_PROXY_EP_SET_OP(ep_get_bcopy);
    UCP_PROXY_EP_SET_OP(ep_get_zcopy);
    UCP_PROXY_EP_SET_OP(ep_am_short);
    UCP_PROXY_EP_SET_OP(ep_am_bcopy);
    UCP_PROXY_EP_SET_OP(ep_am_zcopy);
    UCP_PROXY_EP_SET_OP(ep_atomic_add64);
    UCP_PROXY_EP_SET_OP(ep_atomic_fadd64);
    UCP_PROXY_EP_SET_OP(ep_atomic_swap64);
    UCP_PROXY_EP_SET_OP(ep_atomic_cswap64);
    UCP_PROXY_EP_SET_OP(ep_atomic_add32);
    UCP_PROXY_EP_SET_OP(ep_atomic_fadd32);
    UCP_PROXY_EP_SET_OP(ep_atomic_swap32);
    UCP_PROXY_EP_SET_OP(ep_atomic_cswap32);
    UCP_PROXY_EP_SET_OP(ep_tag_eager_short);
    UCP_PROXY_EP_SET_OP(ep_tag_eager_bcopy);
    UCP_PROXY_EP_SET_OP(ep_tag_eager_zcopy);
    UCP_PROXY_EP_SET_OP(ep_tag_rndv_zcopy);
    UCP_PROXY_EP_SET_OP(ep_tag_rndv_cancel);
    UCP_PROXY_EP_SET_OP(ep_tag_rndv_request);
    UCP_PROXY_EP_SET_OP(ep_pending_add);
    UCP_PROXY_EP_SET_OP(ep_pending_purge);
    UCP_PROXY_EP_SET_OP(ep_flush);
    UCP_PROXY_EP_SET_OP(ep_fence);
    UCP_PROXY_EP_SET_OP(ep_check);
    UCP_PROXY_EP_SET_OP(ep_destroy);
    UCP_PROXY_EP_SET_OP(ep_get_address);
    UCP_PROXY_EP_SET_OP(ep_connect_to_ep);

    self->iface.ops.iface_tag_recv_zcopy     = (void*)ucp_proxy_ep_fatal;
    self->iface.ops.iface_tag_recv_cancel    = (void*)ucp_proxy_ep_fatal;
    self->iface.ops.ep_create                = (void*)ucp_proxy_ep_fatal;
    self->iface.ops.ep_create_connected      = (void*)ucp_proxy_ep_fatal;
    self->iface.ops.iface_flush              = (void*)ucp_proxy_ep_fatal;
    self->iface.ops.iface_fence              = (void*)ucp_proxy_ep_fatal;
    self->iface.ops.iface_progress_enable    = (void*)ucp_proxy_ep_fatal;
    self->iface.ops.iface_progress_disable   = (void*)ucp_proxy_ep_fatal;
    self->iface.ops.iface_progress           = (void*)ucp_proxy_ep_fatal;
    self->iface.ops.iface_event_fd_get       = (void*)ucp_proxy_ep_fatal;
    self->iface.ops.iface_event_arm          = (void*)ucp_proxy_ep_fatal;
    self->iface.ops.iface_close              = (void*)ucp_proxy_ep_fatal;
    self->iface.ops.iface_query              = (void*)ucp_proxy_ep_fatal;
    self->iface.ops.iface_get_device_address = (void*)ucp_proxy_ep_fatal;
    self->iface.ops.iface_get_address        = (void*)ucp_proxy_ep_fatal;
    self->iface.ops.iface_is_reachable       = (void*)ucp_proxy_ep_fatal;

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(ucp_proxy_ep_t)
{
    if ((self->uct_ep != NULL) && self->is_owner) {
        uct_ep_destroy(self->uct_ep);
    }
}

void ucp_proxy_ep_replace(ucp_proxy_ep_t *proxy_ep)
{
    ucp_ep_h ucp_ep = proxy_ep->ucp_ep;
    ucp_lane_index_t lane;

    ucs_assert(proxy_ep->uct_ep != NULL);
    for (lane = 0; lane < ucp_ep_num_lanes(ucp_ep); ++lane) {
        if (ucp_ep->uct_eps[lane] == &proxy_ep->super) {
            ucp_ep->uct_eps[lane] = proxy_ep->uct_ep;
            proxy_ep->uct_ep = NULL;
            break;
        }
    }

    uct_ep_destroy(&proxy_ep->super);
}

void ucp_proxy_ep_set_uct_ep(ucp_proxy_ep_t *proxy_ep, uct_ep_h uct_ep,
                             int is_owner)
{
    proxy_ep->uct_ep   = uct_ep;
    proxy_ep->is_owner = is_owner;
}

UCS_CLASS_DEFINE(ucp_proxy_ep_t, void);
