/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UD_VERBS_H
#define UD_VERBS_H

#include <uct/ib/base/ib_verbs.h>

#include <uct/ib/ud/base/ud_iface.h>
#include <uct/ib/ud/base/ud_ep.h>
#include <uct/ib/ud/base/ud_def.h>


typedef struct {
    uct_ud_ep_t          super;
    uint32_t             dest_qpn;
    struct ibv_ah       *ah;
} uct_ud_verbs_ep_t;

typedef struct {
    uct_ud_iface_t          super;
    struct {
        struct ibv_sge      sge[UCT_IB_MAX_IOV];
        struct ibv_send_wr  wr_inl;
        struct ibv_send_wr  wr_skb;
    } tx;
} uct_ud_verbs_iface_t;

UCS_CLASS_DECLARE(uct_ud_verbs_ep_t, uct_iface_h)

#endif
