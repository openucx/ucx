/**
 * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "gdr_copy_ep.h"
#include "gdr_copy_md.h"
#include "gdr_copy_iface.h"

#include <uct/base/uct_log.h>
#include <ucs/debug/memtrack.h>
#include <ucs/type/class.h>


static UCS_CLASS_INIT_FUNC(uct_gdr_copy_ep_t, uct_iface_t *tl_iface,
                           const uct_device_addr_t *dev_addr,
                           const uct_iface_addr_t *iface_addr)
{
    uct_gdr_copy_iface_t *iface = ucs_derived_of(tl_iface, uct_gdr_copy_iface_t);

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super);

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_gdr_copy_ep_t)
{
}

UCS_CLASS_DEFINE(uct_gdr_copy_ep_t, uct_base_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_gdr_copy_ep_t, uct_ep_t, uct_iface_t*,
                          const uct_device_addr_t *, const uct_iface_addr_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_gdr_copy_ep_t, uct_ep_t);

ucs_status_t uct_gdr_copy_ep_put_short(uct_ep_h tl_ep, const void *buffer,
                                       unsigned length, uint64_t remote_addr,
                                       uct_rkey_t rkey)
{
    uct_gdr_copy_key_t *gdr_copy_key = (uct_gdr_copy_key_t *) rkey;
    size_t bar_offset;
    int ret;

    if (ucs_likely(length)) {
        bar_offset = remote_addr - gdr_copy_key->vaddr;
        ret = gdr_copy_to_bar((gdr_copy_key->bar_ptr + bar_offset), buffer, length);
        if (ret) {
            ucs_error("gdr_copy_to_bar failed. ret:%d", ret);
            return UCS_ERR_IO_ERROR;
        }
    }

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), PUT, SHORT, length);
    ucs_trace_data("PUT_SHORT size %d from %p to %p",
                   length, buffer, (void *)remote_addr);
    return UCS_OK;
}
