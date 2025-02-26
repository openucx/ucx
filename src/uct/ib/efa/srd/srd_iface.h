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


void uct_srd_iface_add_ep(uct_srd_iface_t *iface, uct_srd_ep_t *ep);
void uct_srd_iface_remove_ep(uct_srd_iface_t *iface, uct_srd_ep_t *ep);

ucs_status_t uct_srd_iface_unpack_peer_address(uct_srd_iface_t *iface,
                                               const uct_ib_address_t *ib_addr,
                                               const uct_srd_iface_addr_t *if_addr,
                                               int path_index,
                                               uct_srd_ep_peer_address_t *address);

END_C_DECLS

#endif
