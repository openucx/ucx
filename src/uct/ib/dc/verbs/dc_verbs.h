/**
* Copyright (C) Mellanox Technologies Ltd. 2016-.  ALL RIGHTS RESERVED.

* See file LICENSE for terms.
*/

#ifndef UCT_DC_VERBS_H
#define UCT_DC_VERBS_H

#include <uct/ib/dc/base/dc_iface.h>
#include <uct/ib/dc/base/dc_ep.h>
#include <uct/ib/rc/verbs/rc_verbs_common.h>


#define UCT_DC_VERBS_CHECK_RES_PTR(_iface, _ep) \
    { \
        ucs_status_t status; \
        status = uct_dc_iface_dci_get(_iface, _ep); \
        if (ucs_unlikely(status != UCS_OK)) { \
            return UCS_STATUS_PTR(status); \
        } \
        UCT_RC_CHECK_CQE_RET(&(_iface)->super, _ep, \
                             &(_iface)->tx.dcis[(_ep)->dci].txqp, \
                             UCS_STATUS_PTR(UCS_ERR_NO_RESOURCE)); \
    }


#if IBV_EXP_HW_TM
#  define UCT_DC_VERBS_TM_ENABLE_STR  "TM_ENABLE=n"
#else
#  define UCT_DC_VERBS_TM_ENABLE_STR  ""
#endif


typedef struct uct_dc_verbs_iface_addr {
    uct_dc_iface_addr_t            super;
    uint8_t                        hw_tm;
} UCS_S_PACKED uct_dc_verbs_iface_addr_t;

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
