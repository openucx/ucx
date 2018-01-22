/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "self_ep.h"
#include "self_iface.h"

static UCS_CLASS_INIT_FUNC(uct_self_ep_t, uct_iface_t *tl_iface,
                           const uct_device_addr_t *dev_addr,
                           const uct_iface_addr_t *iface_addr)
{
    uct_self_iface_t *local_iface = 0;

    ucs_trace_func("Creating an EP for loop-back transport self=%p", self);
    local_iface = ucs_derived_of(tl_iface, uct_self_iface_t);
    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &local_iface->super)
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_self_ep_t)
{
    ucs_trace_func("self=%p", self);
}

UCS_CLASS_DEFINE(uct_self_ep_t, uct_base_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_self_ep_t, uct_ep_t, uct_iface_t *,
                          const uct_device_addr_t *, const uct_iface_addr_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_self_ep_t, uct_ep_t);

/**
 * Reserve the buffer and set the descriptor empty for later initialization
 * in case if UCS_ERR_NO_RESOURCE obtained from active message handler
 */
static void UCS_F_ALWAYS_INLINE uct_self_ep_am_reserve_buffer(uct_self_iface_t *self_iface,
                                                              void *desc)
{
    uct_recv_desc(desc) = &self_iface->release_desc;
}

ucs_status_t uct_self_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t header,
                                  const void *payload, unsigned length)
{
    ucs_status_t status;
    uct_self_iface_t *self_iface = 0;
    uct_self_ep_t *self_ep = 0;
    void *desc = 0, *p_data = 0;
    unsigned total_length = 0;
    uct_recv_desc_t *cur_desc;

    self_ep = ucs_derived_of(tl_ep, uct_self_ep_t);
    self_iface = ucs_derived_of(self_ep->super.super.iface, uct_self_iface_t);
    total_length = length + sizeof(header);

    /* Send part */
    UCT_CHECK_AM_ID(id);
    UCT_CHECK_LENGTH(total_length, 0, self_iface->data_length, "am_short");
    UCT_TL_IFACE_GET_TX_DESC(&self_iface->super, &self_iface->msg_desc_mp,
                             cur_desc, return UCS_ERR_NO_MEMORY);

    desc = cur_desc + 1;
    p_data = desc + self_iface->rx_headroom;
    *(typeof(header)*) p_data = header;
    memcpy(p_data + sizeof(header), payload, length);

    UCT_TL_EP_STAT_OP(&self_ep->super, AM, SHORT, total_length);
    uct_iface_trace_am(&self_iface->super, UCT_AM_TRACE_TYPE_SEND, id, p_data,
                       total_length, "TX: AM_SHORT");

    /* Receive part */
    uct_iface_trace_am(&self_iface->super, UCT_AM_TRACE_TYPE_RECV, id, p_data,
                       total_length, "RX: AM_SHORT");
    status = uct_iface_invoke_am(&self_iface->super, id, p_data, total_length,
                                 UCT_CB_PARAM_FLAG_DESC);

    if (ucs_unlikely(UCS_INPROGRESS == status)) {
        uct_self_ep_am_reserve_buffer(self_iface, desc);
        status = UCS_OK;
    } else {
        UCT_TL_IFACE_PUT_DESC(cur_desc);
    }

    return status;
}

ssize_t uct_self_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                             uct_pack_callback_t pack_cb, void *arg,
                             unsigned flags)
{
    ucs_status_t status;
    uct_self_iface_t *self_iface = 0;
    uct_self_ep_t *self_ep = 0;
    void *desc = 0, *payload = 0;
    ssize_t length = 0;
    uct_recv_desc_t *cur_desc;

    self_ep = ucs_derived_of(tl_ep, uct_self_ep_t);
    self_iface = ucs_derived_of(self_ep->super.super.iface, uct_self_iface_t);

    /* Send part */
    UCT_CHECK_AM_ID(id);
    UCT_TL_IFACE_GET_TX_DESC(&self_iface->super, &self_iface->msg_desc_mp,
                             cur_desc, return UCS_ERR_NO_MEMORY);

    desc = cur_desc + 1;
    payload = desc + self_iface->rx_headroom;
    length = pack_cb(payload, arg);

    UCT_CHECK_LENGTH(length, 0, self_iface->data_length, "am_bcopy");
    UCT_TL_EP_STAT_OP(&self_ep->super, AM, BCOPY, length);
    uct_iface_trace_am(&self_iface->super, UCT_AM_TRACE_TYPE_SEND, id, payload,
                       length, "TX: AM_BCOPY");

    /* Receive part */
    uct_iface_trace_am(&self_iface->super, UCT_AM_TRACE_TYPE_RECV, id, payload,
                       length, "RX: AM_BCOPY");
    status = uct_iface_invoke_am(&self_iface->super, id, payload, length,
                                 UCT_CB_PARAM_FLAG_DESC);

    if (ucs_unlikely(UCS_INPROGRESS == status)) {
        uct_self_ep_am_reserve_buffer(self_iface, desc);
        status = UCS_OK;
    } else {
        UCT_TL_IFACE_PUT_DESC(cur_desc);
    }

    return length;
}
