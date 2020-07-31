/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
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
    uint32_t                          dest_qpn;
    struct ibv_ah                     *ah;
} uct_ud_verbs_ep_peer_address_t;


typedef struct {
    uct_ud_ep_t                       super;
    uct_ud_verbs_ep_peer_address_t    peer_address;
} uct_ud_verbs_ep_t;


typedef struct {
    uct_ud_iface_t                    super;
    struct {
        struct ibv_sge                sge[UCT_IB_MAX_IOV];
        struct ibv_send_wr            wr_inl;
        struct ibv_send_wr            wr_skb;
        uint16_t                      send_sn;
        uint16_t                      comp_sn;
    } tx;
    struct {
        size_t                        max_send_sge;
    } config;
} uct_ud_verbs_iface_t;


UCS_CLASS_DECLARE(uct_ud_verbs_ep_t, const uct_ep_params_t *)


ucs_status_t uct_ud_verbs_qp_max_send_sge(uct_ud_verbs_iface_t *iface,
                                          size_t *max_send_sge);

#endif
