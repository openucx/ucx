/*
 * Copyright (C) Advanced Micro Devices, Inc. 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "musa_copy_ep.h"
#include "musa_copy_iface.h"
#include "musa_base.h"

#include <uct/base/uct_iov.inl>
#include <ucs/type/class.h>

static UCS_CLASS_INIT_FUNC(uct_musa_copy_ep_t, const uct_ep_params_t *params)
{
    uct_musa_copy_iface_t *iface =
            ucs_derived_of(params->iface, uct_musa_copy_iface_t);

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_musa_copy_ep_t)
{
}

UCS_CLASS_DEFINE(uct_musa_copy_ep_t, uct_base_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_musa_copy_ep_t, uct_ep_t,
                          const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_musa_copy_ep_t, uct_ep_t);

static ucs_status_t
uct_musa_copy_ep_copy(uct_ep_h tl_ep, const void *src, void *dst, size_t length)
{
    uct_musa_copy_iface_t *iface =
            ucs_derived_of(tl_ep->iface, uct_musa_copy_iface_t);

    return uct_musa_base_copy(src, dst, length, iface->device);
}

ucs_status_t uct_musa_copy_ep_put_short(uct_ep_h tl_ep, const void *buffer,
                                         unsigned length, uint64_t remote_addr,
                                         uct_rkey_t rkey)
{
    ucs_status_t status;

    (void)rkey;
    status = uct_musa_copy_ep_copy(tl_ep, buffer, (void *)remote_addr, length);
    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), PUT, SHORT, length);
    return status;
}

ucs_status_t uct_musa_copy_ep_get_short(uct_ep_h tl_ep, void *buffer,
                                         unsigned length, uint64_t remote_addr,
                                         uct_rkey_t rkey)
{
    ucs_status_t status;

    (void)rkey;
    status = uct_musa_copy_ep_copy(tl_ep, (void *)remote_addr, buffer, length);
    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), GET, SHORT, length);
    return status;
}

static ucs_status_t
uct_musa_copy_ep_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                       uint64_t remote_addr, int is_put)
{
    size_t length;

    if ((iovcnt != 1) || (iov[0].count != 1)) {
        return UCS_ERR_UNSUPPORTED;
    }

    length = uct_iov_get_length(iov);
    if (is_put) {
        return uct_musa_copy_ep_copy(tl_ep, iov[0].buffer,
                                     (void *)remote_addr, length);
    }

    return uct_musa_copy_ep_copy(tl_ep, (void *)remote_addr, iov[0].buffer,
                                 length);
}

ucs_status_t uct_musa_copy_ep_put_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov,
                                         size_t iovcnt, uint64_t remote_addr,
                                         uct_rkey_t rkey,
                                         uct_completion_t *comp)
{
    (void)rkey;
    (void)comp;
    return uct_musa_copy_ep_zcopy(tl_ep, iov, iovcnt, remote_addr, 1);
}

ucs_status_t uct_musa_copy_ep_get_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov,
                                         size_t iovcnt, uint64_t remote_addr,
                                         uct_rkey_t rkey,
                                         uct_completion_t *comp)
{
    (void)rkey;
    (void)comp;
    return uct_musa_copy_ep_zcopy(tl_ep, iov, iovcnt, remote_addr, 0);
}
