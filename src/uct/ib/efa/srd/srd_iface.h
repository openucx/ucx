/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_SRD_IFACE_H
#define UCT_SRD_IFACE_H

#include "srd_def.h"

#include <uct/ib/ud/base/ud_iface_common.h>

#include <ucs/datastruct/ptr_array.h>
#include <ucs/datastruct/conn_match.h>

BEGIN_C_DECLS


typedef struct uct_srd_iface_config {
    uct_ib_iface_config_t        super;
    uct_ud_iface_common_config_t ud_common;

    struct {
        size_t                   max_get_zcopy;
    } tx;
} uct_srd_iface_config_t;


typedef struct uct_srd_iface {
    uct_ib_iface_t         super;
    struct ibv_qp          *qp;
#ifdef HAVE_DECL_EFA_DV_RDMA_READ
    struct ibv_qp_ex       *qp_ex;
#endif
    ucs_ptr_array_t        eps;

    struct {
        unsigned           available;
    } rx;

    struct {
        int32_t            available;
        ucs_arbiter_t      pending_q;
        struct ibv_sge     sge[UCT_IB_MAX_IOV];
        struct ibv_send_wr wr_inl;
        struct ibv_send_wr wr_desc;
    } tx;

    struct {
        unsigned           tx_qp_len;
        unsigned           max_inline;
        size_t             max_send_sge;
        size_t             max_get_zcopy;
    } config;
} uct_srd_iface_t;


END_C_DECLS

#endif
