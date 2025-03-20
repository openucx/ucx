/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_SRD_IFACE_H
#define UCT_SRD_IFACE_H

#include "srd_def.h"
#include "srd_ep.h"

#include <uct/ib/ud/base/ud_iface_common.h>

#include <ucs/datastruct/ptr_array.h>

BEGIN_C_DECLS


typedef struct uct_srd_iface_config {
    uct_ib_iface_config_t        super;
    uct_ud_iface_common_config_t ud_common;

    struct {
        size_t                   max_get_zcopy;
    } tx;
} uct_srd_iface_config_t;


typedef struct uct_srd_iface {
    uct_ib_iface_t             super;
    struct ibv_qp              *qp;
#ifdef HAVE_DECL_EFA_DV_RDMA_READ
    struct ibv_qp_ex           *qp_ex;
#endif

    struct {
        unsigned               available;
        ucs_mpool_t            mp;
    } rx;

    struct {
        int32_t                available;
        ucs_arbiter_t          pending_q;
        struct ibv_sge         sge[UCT_IB_MAX_IOV];
        struct ibv_send_wr     wr_inl;
        struct ibv_send_wr     wr_desc;
        ucs_mpool_t            send_op_mp;
        uct_srd_am_short_hdr_t am_inl_hdr;
        ucs_list_link_t        outstanding_list;
    } tx;

    struct {
        unsigned               tx_qp_len;
        unsigned               max_inline;
        size_t                 max_send_sge;
        size_t                 max_get_zcopy;
    } config;
} uct_srd_iface_t;


#if ENABLE_PARAMS_CHECK
#define UCT_SRD_CHECK_LENGTH_MTU(_iface, _tx_len, _msg) \
     do { \
         int _mtu; \
         _mtu =  uct_ib_mtu_value(uct_ib_iface_port_attr(&(_iface)->super)->active_mtu); \
         UCT_CHECK_LENGTH(_tx_len, 0, _mtu, _msg); \
     } while(0);

#else
#define UCT_SRD_CHECK_LENGTH_MTU(_iface, _tx_len, _msg)
#endif

#define UCT_SRD_CHECK_AM_LEN(_iface, _id, _data_len, _max_len, _msg) \
    UCT_CHECK_LENGTH(sizeof(uct_srd_hdr_t) + (_data_len), 0, _max_len, _msg); \
    UCT_SRD_CHECK_LENGTH_MTU(_iface, sizeof(uct_srd_hdr_t) + (_data_len), _msg"_mtu");

#define UCT_SRD_CHECK_AM_SHORT(_iface, _id, _hdr_len, _data_len) \
    UCT_CHECK_AM_ID(_id); \
    UCT_SRD_CHECK_AM_LEN(_iface, _id, (_hdr_len) + (_data_len), \
                         (_iface)->config.max_inline, "am_short");


static UCS_F_ALWAYS_INLINE int
uct_srd_iface_can_tx(const uct_srd_iface_t *iface)
{
    return iface->tx.available > 0;
}


static UCS_F_ALWAYS_INLINE uct_srd_send_op_t *
uct_srd_iface_get_send_op(uct_srd_iface_t *iface)
{
    uct_srd_send_op_t *send_op;

    if (ucs_unlikely(!uct_srd_iface_can_tx(iface))) {
        return NULL;
    }

    send_op = ucs_mpool_get(&iface->tx.send_op_mp);
    if (ucs_unlikely(send_op == NULL)) {
        ucs_trace_data("iface=%p out of tx send_op descs", iface);
        UCT_TL_IFACE_STAT_TX_NO_DESC(&iface->super.super);
    }

    return send_op;
}


END_C_DECLS

#endif
