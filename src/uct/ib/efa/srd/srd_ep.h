/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_SRD_EP_H
#define UCT_SRD_EP_H

#include "srd_def.h"


typedef struct uct_srd_ep {
    uct_base_ep_t       super;
    uint64_t            ep_uuid;       /* Random EP identifier */
    uint32_t            dest_qpn;      /* Remote QP */
    uint32_t            inflight;      /* Entries outstanding list */
    struct ibv_ah       *ah;           /* Remote peer */
    int                 ah_added;      /* true if remote has added our AH */
    uct_srd_psn_t       psn;           /* Next PSN to send */
    uint8_t             path_index;
    ucs_arbiter_group_t pending_group; /* Queue of pending requests */
} uct_srd_ep_t;


UCS_CLASS_DECLARE_NEW_FUNC(uct_srd_ep_t, uct_ep_t, const uct_ep_params_t*);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_srd_ep_t, uct_ep_t);


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

void uct_srd_ep_send_op_completion(uct_srd_send_op_t *send_op);

ucs_status_t
uct_srd_ep_pending_add(uct_ep_h tl_ep, uct_pending_req_t *req, unsigned flags);
void uct_srd_ep_pending_purge(uct_ep_h ep, uct_pending_purge_callback_t cb,
                              void *arg);
ucs_arbiter_cb_result_t
uct_srd_ep_do_pending(ucs_arbiter_t *arbiter, ucs_arbiter_group_t *group,
                      ucs_arbiter_elem_t *elem, void *arg);

#endif
