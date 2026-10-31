/*
 * Copyright (C) MThreads 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "musa_copy_ep.h"
#include "musa_copy_iface.h"

#include <uct/musa/base/musa_base.h>

#include <uct/base/uct_iov.inl>
#include <ucs/type/class.h>

static UCS_CLASS_INIT_FUNC(uct_musa_copy_ep_t, const uct_ep_params_t *params)
{
    uct_musa_copy_iface_t *iface = ucs_derived_of(params->iface,
                                                  uct_musa_copy_iface_t);

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
uct_musa_copy_ep_copy(uct_ep_h tl_ep, const void *src, void *dst,
                      size_t length)
{
    uct_musa_copy_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                  uct_musa_copy_iface_t);

    return uct_musa_base_copy(src, dst, length, iface->device);
}

ucs_status_t uct_musa_copy_ep_put_short(uct_ep_h tl_ep, const void *buffer,
                                        unsigned length, uint64_t remote_addr,
                                        uct_rkey_t rkey)
{
    ucs_status_t status;

    status = uct_musa_copy_ep_copy(tl_ep, buffer, (void*)remote_addr, length);
    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), PUT, SHORT, length);
    return status;
}

ucs_status_t uct_musa_copy_ep_get_short(uct_ep_h tl_ep, void *buffer,
                                        unsigned length, uint64_t remote_addr,
                                        uct_rkey_t rkey)
{
    ucs_status_t status;

    status = uct_musa_copy_ep_copy(tl_ep, (void*)remote_addr, buffer, length);
    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), GET, SHORT, length);
    return status;
}

static ucs_status_t
uct_musa_copy_ep_get_copy_context(uct_musa_copy_iface_t *iface,
                                  const void *src, void *dst, size_t length,
                                  ucs_memory_type_t *src_mem_type_p,
                                  ucs_memory_type_t *dst_mem_type_p,
                                  int *device_p,
                                  uct_musa_pointer_info_t *src_info_p,
                                  uct_musa_pointer_info_t *dst_info_p)
{
    ucs_status_t status;

    status = uct_musa_base_query_copy_pointer(src, length, src_info_p);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_musa_base_query_copy_pointer(dst, length, dst_info_p);
    if (status != UCS_OK) {
        return status;
    }

    if ((src_info_p->kind == UCT_MUSA_POINTER_MANAGED) ||
        (dst_info_p->kind == UCT_MUSA_POINTER_MANAGED)) {
        return UCS_ERR_UNSUPPORTED;
    }

    *src_mem_type_p = (src_info_p->kind == UCT_MUSA_POINTER_DEVICE) ?
                      UCS_MEMORY_TYPE_MUSA : UCS_MEMORY_TYPE_HOST;
    *dst_mem_type_p = (dst_info_p->kind == UCT_MUSA_POINTER_DEVICE) ?
                      UCS_MEMORY_TYPE_MUSA : UCS_MEMORY_TYPE_HOST;

    if (src_info_p->device >= 0) {
        *device_p = src_info_p->device;
    } else if (dst_info_p->device >= 0) {
        *device_p = dst_info_p->device;
    } else {
        *device_p = iface->device;
    }

    return UCS_OK;
}

static ucs_status_t
uct_musa_copy_ep_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                       uint64_t remote_addr, uct_completion_t *comp,
                       int is_put)
{
    uct_musa_copy_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                  uct_musa_copy_iface_t);
    uct_musa_copy_device_rsc_t *device_rsc;
    uct_musa_copy_queue_desc_t *queue_desc;
    ucs_memory_type_t src_mem_type;
    ucs_memory_type_t dst_mem_type;
    uct_musa_pointer_info_t src_info;
    uct_musa_pointer_info_t dst_info;
    void *src;
    void *dst;
    size_t length;
    int device;
    ucs_status_t status;

    if ((iovcnt != 1) || (iov[0].count != 1)) {
        return UCS_ERR_UNSUPPORTED;
    }

    length = uct_iov_get_length(iov);
    if (length == 0) {
        return UCS_OK;
    }

    src    = is_put ? iov[0].buffer : (void*)remote_addr;
    dst    = is_put ? (void*)remote_addr : iov[0].buffer;

    status = uct_musa_copy_ep_get_copy_context(iface, src, dst, length,
                                               &src_mem_type, &dst_mem_type,
                                               &device, &src_info, &dst_info);
    if (status != UCS_OK) {
        return status;
    }

    if ((src_mem_type == UCS_MEMORY_TYPE_HOST) &&
        (dst_mem_type == UCS_MEMORY_TYPE_HOST)) {
        return uct_musa_copy_ep_copy(tl_ep, src, dst, length);
    }

    device_rsc = uct_musa_copy_iface_get_device_rsc(iface, device);
    if (device_rsc == NULL) {
        return UCS_ERR_NO_DEVICE;
    }

    queue_desc = &device_rsc->queue_desc[src_mem_type][dst_mem_type];
    status = uct_musa_copy_iface_get_stream(device_rsc, queue_desc);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_musa_base_copy_async(src, dst, length, device,
                                      queue_desc->stream, &src_info,
                                      &dst_info);
    if (status != UCS_OK) {
        return status;
    }

    return uct_musa_copy_iface_post_event(device_rsc, queue_desc, comp, NULL);
}

ucs_status_t uct_musa_copy_ep_put_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov,
                                        size_t iovcnt, uint64_t remote_addr,
                                        uct_rkey_t rkey, uct_completion_t *comp)
{
    ucs_status_t status;

    status = uct_musa_copy_ep_zcopy(tl_ep, iov, iovcnt, remote_addr, comp, 1);
    if (!UCS_STATUS_IS_ERR(status)) {
        UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), PUT, ZCOPY,
                          uct_iov_total_length(iov, iovcnt));
    }
    return status;
}

ucs_status_t uct_musa_copy_ep_get_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov,
                                        size_t iovcnt, uint64_t remote_addr,
                                        uct_rkey_t rkey, uct_completion_t *comp)
{
    ucs_status_t status;

    status = uct_musa_copy_ep_zcopy(tl_ep, iov, iovcnt, remote_addr, comp, 0);
    if (!UCS_STATUS_IS_ERR(status)) {
        UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), GET, ZCOPY,
                          uct_iov_total_length(iov, iovcnt));
    }
    return status;
}

ucs_status_t uct_musa_copy_ep_flush(uct_ep_h tl_ep, unsigned flags,
                                    uct_completion_t *comp)
{
    return uct_musa_copy_iface_flush(tl_ep->iface, flags, comp);
}
