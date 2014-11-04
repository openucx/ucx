/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_IB_VERBS_H
#define UCT_IB_VERBS_H

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <infiniband/verbs.h>
#ifdef HAVE_VERBS_EXP_H
#include <infiniband/verbs_exp.h>
#endif


#ifndef HAVE_VERBS_EXP_H
#  define IBV_EXP_SEND_INLINE              IBV_SEND_INLINE
#  define IBV_EXP_SEND_SIGNALED            IBV_SEND_SIGNALED
#  define IBV_EXP_SEND_SOLICITED           IBV_SEND_SOLICITED
#  define IBV_EXP_SEND_FENCE               IBV_SEND_FENCE
#  define IBV_EXP_QP_STATE                 IBV_QP_STATE
#  define IBV_EXP_QP_PKEY_INDEX            IBV_QP_PKEY_INDEX
#  define IBV_EXP_QP_PORT                  IBV_QP_PORT
#  define IBV_EXP_QP_PATH_MTU              IBV_QP_PATH_MTU
#  define IBV_EXP_QP_TIMEOUT               IBV_QP_TIMEOUT
#  define IBV_EXP_QP_AV                    IBV_QP_AV
#  define IBV_EXP_QP_RETRY_CNT             IBV_QP_RETRY_CNT
#  define IBV_EXP_QP_RNR_RETRY             IBV_QP_RNR_RETRY
#  define IBV_EXP_QP_MAX_QP_RD_ATOMIC      IBV_QP_MAX_QP_RD_ATOMIC
#  define IBV_EXP_ACCESS_REMOTE_WRITE      IBV_ACCESS_REMOTE_WRITE
#  define IBV_EXP_ACCESS_REMOTE_READ       IBV_ACCESS_REMOTE_READ
#  define IBV_EXP_ACCESS_REMOTE_ATOMIC     IBV_ACCESS_REMOTE_ATOMIC
#  define exp_send_flags                   send_flags
#  define IBV_EXP_ACCESS_ALLOCATE_MR       IBV_ACCESS_ALLOCATE_MR
#  define IBV_EXP_ATOMIC_HCA               IBV_ATOMIC_HCA
#  define ibv_exp_reg_shared_mr            ibv_reg_shared_mr_ex
#  define ibv_exp_reg_shared_mr_in         ibv_reg_shared_mr_in
#  define ibv_exp_query_device             ibv_query_device
#  define ibv_exp_device_attr              ibv_device_attr
#  define exp_atomic_cap                   atomic_cap
#  define ibv_exp_modify_cq                ibv_modify_cq
#  define ibv_exp_cq_attr                  ibv_cq_attr
#  define IBV_EXP_CQ_ATTR_CQ_CAP_FLAGS     IBV_CQ_ATTR_CQ_CAP_FLAGS
#  define IBV_EXP_CQ_IGNORE_OVERRUN        IBV_CQ_ATTR_CQ_CAP_FLAGS
#  define IBV_EXP_CQ_CAP_FLAGS             IBV_CQ_CAP_FLAGS
#  define ibv_exp_send_wr                  ibv_send_wr
#  define exp_opcode                       opcode
#  define ibv_exp_post_send                ibv_post_send
#  define IBV_EXP_WR_NOP                   IBV_WR_NOP
#  define IBV_EXP_WR_SEND                  IBV_WR_SEND
#  define IBV_EXP_WR_RDMA_WRITE            IBV_WR_RDMA_WRITE
#  define IBV_EXP_WR_RDMA_READ             IBV_WR_RDMA_READ
#  define IBV_EXP_WR_ATOMIC_FETCH_AND_ADD  IBV_WR_ATOMIC_FETCH_AND_ADD
#  define IBV_EXP_WR_ATOMIC_CMP_AND_SWP    IBV_WR_ATOMIC_CMP_AND_SWP
#  define ibv_exp_qp_init_attr             ibv_qp_init_attr
#  define ibv_exp_create_qp                ibv_create_qp

struct ibv_exp_reg_mr_in {
    struct ibv_pd *pd;
    void *addr;
    size_t length;
    int exp_access;
    uint32_t comp_mask;
};

static inline struct ibv_mr *ibv_exp_reg_mr(struct ibv_exp_reg_mr_in *in)
{
    return ibv_reg_mr(in->pd, in->addr, in->length, in->exp_access);
}

#  define IBV_IS_MPAGES_AVAIL(_attr)                ((_attr)->exp_device_cap_flags & IBV_EXP_DEVICE_MR_ALLOCATE)
#  define IBV_EXP_REG_MR_FLAGS(_f, _e)              ((_f) | (_e))
#  define IBV_SHARED_MR_ACCESS_FLAGS(_shared_mr)    ((_shared_mr)->exp_access)
#  define IBV_EXP_DEVICE_ATTR_SET_COMP_MASK(_attr)
#  define IBV_EXP_PORT_ATTR_SET_COMP_MASK(_attr)
#else
#  define IBV_IS_MPAGES_AVAIL(_attr)                ((_attr)->device_cap_flags2 & IBV_EXP_DEVICE_MR_ALLOCATE)
#  define IBV_EXP_REG_MR_FLAGS(_f, _e)              (_f) , (_e)
#  define IBV_SHARED_MR_ACCESS_FLAGS(_shared_mr)    ((_shared_mr)->access)
#  define IBV_EXP_DEVICE_ATTR_SET_COMP_MASK(_attr)  (_attr)->comp_mask = (IBV_EXP_DEVICE_ATTR_RESERVED - 1)
#  define IBV_EXP_PORT_ATTR_SET_COMP_MASK(_attr)    (_attr)->comp_mask = 0
#endif

#endif
