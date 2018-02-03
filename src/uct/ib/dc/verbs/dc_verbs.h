/**
* Copyright (C) Mellanox Technologies Ltd. 2016-.  ALL RIGHTS RESERVED.

* See file LICENSE for terms.
*/

#ifndef UCT_DC_VERBS_H
#define UCT_DC_VERBS_H

#include <uct/ib/dc/base/dc_iface.h>
#include <uct/ib/dc/base/dc_ep.h>
#include <uct/ib/rc/verbs/rc_verbs_common.h>


typedef struct uct_dc_verbs_iface_config {
    uct_dc_iface_config_t              super;
    uct_rc_verbs_iface_common_config_t verbs_common;
} uct_dc_verbs_iface_config_t;


typedef struct uct_dc_verbs_iface {
    uct_dc_iface_t                 super;
    struct ibv_exp_send_wr         inl_am_wr;
    struct ibv_exp_send_wr         inl_rwrite_wr;
    uct_rc_verbs_iface_common_t    verbs_common;
    uct_rc_verbs_txcnt_t           dcis_txcnt[UCT_DC_IFACE_MAX_DCIS];
} uct_dc_verbs_iface_t;


typedef struct uct_dc_verbs_ep {
    uct_dc_ep_t                    super;
    uint32_t                       dest_qpn;
    struct ibv_ah                  *ah;
} uct_dc_verbs_ep_t;


#endif
