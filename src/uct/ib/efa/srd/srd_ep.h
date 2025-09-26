/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_SRD_EP_H
#define UCT_SRD_EP_H

#include "srd_def.h"


typedef enum uct_srd_ep_flag {
    UCT_SRD_EP_FLAG_CANCELED    = UCS_BIT(0), /* Endpoint was flush canceled */
    UCT_SRD_EP_FLAG_AH_ADDED    = UCS_BIT(1), /* Remote has added AH */
    UCT_SRD_EP_FLAG_FENCE       = UCS_BIT(2), /* EP fence operation requested */
    UCT_SRD_EP_FLAG_ERR_HANDLER_INVOKED
                                = UCS_BIT(3), /* EP error handler was invoked */
    UCT_SRD_EP_FLAG_RMA         = UCS_BIT(4)  /* Endpoint has seen RMA post */
} uct_srd_ep_flag_t;


typedef struct uct_srd_ep {
    uct_base_ep_t       super;
    unsigned            flags;            /* Endpoint state tracking */
    uint64_t            ep_uuid;          /* Random EP identifier */
    uint32_t            dest_qpn;         /* Remote QP */
    struct ibv_ah       *ah;              /* Remote peer */
    uct_srd_psn_t       psn;              /* Next PSN to send */
    uint8_t             path_index;
    ucs_arbiter_group_t pending_group;    /* Queue of pending requests */
    ucs_list_link_t     outstanding_list; /* Ordered outstanding list */
} uct_srd_ep_t;


UCS_CLASS_DECLARE_NEW_FUNC(uct_srd_ep_t, uct_ep_t, const uct_ep_params_t*);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_srd_ep_t, uct_ep_t);


int uct_srd_ep_is_connected(const uct_ep_h tl_ep,
                            const uct_ep_is_connected_params_t *params);
ucs_status_t uct_srd_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                                 const void *buffer, unsigned length);
ucs_status_t uct_srd_ep_am_short_iov(uct_ep_h tl_ep, uint8_t id,
                                     const uct_iov_t *iov, size_t iovcnt);
ssize_t uct_srd_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                            uct_pack_callback_t pack_cb, void *arg,
                            unsigned flags);
ucs_status_t uct_srd_ep_get_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov,
                                  size_t iovcnt, uint64_t remote_addr,
                                  uct_rkey_t rkey, uct_completion_t *comp);
ucs_status_t uct_srd_ep_get_bcopy(uct_ep_h tl_ep,
                                  uct_unpack_callback_t unpack_cb, void *arg,
                                  size_t length, uint64_t remote_addr,
                                  uct_rkey_t rkey, uct_completion_t *comp);
ucs_status_t uct_srd_ep_am_zcopy(uct_ep_h tl_ep, uint8_t id, const void *header,
                                 unsigned header_length, const uct_iov_t *iov,
                                 size_t iovcnt, unsigned flags,
                                 uct_completion_t *comp);
ucs_status_t uct_srd_ep_put_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov,
                                  size_t iovcnt, uint64_t remote_addr,
                                  uct_rkey_t rkey, uct_completion_t *comp);
ssize_t uct_srd_ep_put_bcopy(uct_ep_h tl_ep, uct_pack_callback_t pack_cb,
                             void *arg, uint64_t remote_addr, uct_rkey_t rkey);
ucs_status_t uct_srd_ep_fence(uct_ep_h ep, unsigned flags);
ucs_status_t uct_srd_ep_flush(uct_ep_h ep_h, unsigned flags,
                              uct_completion_t *comp);
void uct_srd_ep_send_op_purge(uct_srd_ep_t *ep);

void uct_srd_ep_send_op_completion(uct_srd_send_op_t *send_op);

ucs_status_t
uct_srd_ep_pending_add(uct_ep_h tl_ep, uct_pending_req_t *req, unsigned flags);
void uct_srd_ep_pending_purge(uct_ep_h ep, uct_pending_purge_callback_t cb,
                              void *arg);
ucs_arbiter_cb_result_t
uct_srd_ep_do_pending(ucs_arbiter_t *arbiter, ucs_arbiter_group_t *group,
                      ucs_arbiter_elem_t *elem, void *arg);

#endif
