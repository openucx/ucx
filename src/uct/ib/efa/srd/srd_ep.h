/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_SRD_EP_H
#define UCT_SRD_EP_H

#include "srd_def.h"


#define UCT_SRD_INITIAL_PSN   1
#define UCT_SRD_EP_NULL_ID    UCS_MASK(24)
#define UCT_SRD_SEND_OP_ALIGN UCS_SYS_CACHE_LINE_SIZE


typedef struct uct_srd_send_op      uct_srd_send_op_t;


typedef struct uct_srd_ep {
    uct_base_ep_t   super;
    uint64_t        ep_uuid;          /* Random EP identifier */
    uint32_t        dest_qpn;         /* Remote QP */
    uint32_t        inflight;         /* Entries outstanding list */
    struct ibv_ah   *ah;              /* Remote peer */
    uct_srd_psn_t   psn;              /* Next PSN to send */
    uint8_t         path_index;
} uct_srd_ep_t;


/*
 * Send descriptor used when receiving TX CQE.
 */
struct uct_srd_send_op {
    ucs_list_link_t                  list;         /* Link in ep outstanding send list */
    uct_srd_ep_t                     *ep;          /* Sender EP */
} UCS_V_ALIGNED(UCT_SRD_SEND_OP_ALIGN);


UCS_CLASS_DECLARE_NEW_FUNC(uct_srd_ep_t, uct_ep_t, const uct_ep_params_t*);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_srd_ep_t, uct_ep_t);


ucs_status_t uct_srd_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                                 const void *buffer, unsigned length);

void uct_srd_ep_send_op_completion(uct_srd_send_op_t *send_op);

#endif
