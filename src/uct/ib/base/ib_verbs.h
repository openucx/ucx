/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
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

#include <errno.h>

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
#  define ibv_exp_reg_shared_mr            ibv_reg_shared_mr_ex
#  define ibv_exp_reg_shared_mr_in         ibv_reg_shared_mr_in
#  define ibv_exp_query_device             ibv_query_device
#  define ibv_exp_device_attr              ibv_device_attr
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
#  define ibv_exp_port_attr                ibv_port_attr
#  define ibv_exp_query_port               ibv_query_port
#  define exp_device_cap_flags             device_cap_flags
#  define ibv_exp_create_qp                ibv_create_qp

#  define IBV_SHARED_MR_ACCESS_FLAGS(_shared_mr)    ((_shared_mr)->exp_access)
#  define IBV_EXP_DEVICE_ATTR_SET_COMP_MASK(_attr)
#  define IBV_EXP_PORT_ATTR_SET_COMP_MASK(_attr)

#else
#  define IBV_SHARED_MR_ACCESS_FLAGS(_shared_mr)    ((_shared_mr)->access)
#  define IBV_EXP_DEVICE_ATTR_SET_COMP_MASK(_attr)  (_attr)->comp_mask = (IBV_EXP_DEVICE_ATTR_RESERVED - 1)
#  define IBV_EXP_PORT_ATTR_SET_COMP_MASK(_attr)    (_attr)->comp_mask = 0
#endif /* HAVE_VERBS_EXP_H */


/*
 * Contiguous pages support
 */
#if HAVE_DECL_IBV_EXP_DEVICE_MR_ALLOCATE
#  define IBV_EXP_HAVE_CONTIG_PAGES(_attr)          ((_attr)->exp_device_cap_flags & IBV_EXP_DEVICE_MR_ALLOCATE)
#else
#  define IBV_EXP_HAVE_CONTIG_PAGES(_attr)         0
#endif


/*
 * On-demand paging support
 */
#if HAVE_STRUCT_IBV_EXP_DEVICE_ATTR_ODP_CAPS
#  define IBV_EXP_HAVE_ODP(_attr)                   ((_attr)->odp_caps.general_odp_caps & IBV_EXP_ODP_SUPPORT)
#  define IBV_EXP_ODP_CAPS(_attr, _xport)           ((_attr)->odp_caps.per_transport_caps._xport##_odp_caps)
#else
#  define IBV_EXP_HAVE_ODP(_attr)                   0
#  define IBV_EXP_ODP_CAPS(_attr, _xport)           0
#endif

#if !HAVE_DECL_IBV_EXP_ACCESS_ON_DEMAND
#  define IBV_EXP_ACCESS_ON_DEMAND                  0
#endif

#if !HAVE_DECL_IBV_EXP_PREFETCH_WRITE_ACCESS
#  define IBV_EXP_PREFETCH_WRITE_ACCESS             IBV_EXP_ACCESS_LOCAL_WRITE
#endif

/*
 * DC support
 */
#if HAVE_DECL_IBV_EXP_DEVICE_DC_TRANSPORT && HAVE_STRUCT_IBV_EXP_DEVICE_ATTR_EXP_DEVICE_CAP_FLAGS
#  define IBV_DEVICE_HAS_DC(_attr)                  ((_attr)->exp_device_cap_flags & IBV_EXP_DEVICE_DC_TRANSPORT)
#else
#  define IBV_DEVICE_HAS_DC(_attr)                  0
#endif /* HAVE_DECL_IBV_EXP_DEVICE_DC_TRANSPORT */


/*
 * NOP support
 */
#if HAVE_DECL_IBV_EXP_WR_NOP
#  define IBV_DEVICE_HAS_NOP(_attr)                 ((_attr)->exp_device_cap_flags & IBV_EXP_DEVICE_NOP)
#else
#  define IBV_DEVICE_HAS_NOP(_attr)                 0
#endif /* HAVE_DECL_IBV_EXP_WR_NOP */

/*
 * Adaptive Routing support
 */
#if HAVE_DECL_IBV_EXP_QP_OOO_RW_DATA_PLACEMENT
#  define UCX_IB_DEV_IS_OOO_SUPPORTED(_attr, _transport)  \
    (((_attr)->comp_mask & IBV_EXP_DEVICE_ATTR_OOO_CAPS) && \
     ((_attr)->ooo_caps._transport##_caps & IBV_EXP_OOO_SUPPORT_RW_DATA_PLACEMENT))
#else
#  define UCX_IB_DEV_IS_OOO_SUPPORTED(_ibdev, _transport)   0
#endif


/*
 * Safe setenv
 */
#if HAVE_DECL_IBV_EXP_SETENV
#  define ibv_exp_unsetenv(_c, _n)                  0
#else
#  define ibv_exp_setenv(_c, _n, _v, _o)            setenv(_n, _v, _o)
#  define ibv_exp_unsetenv(_c, _n)                  unsetenv(_n)
#endif


/*
 * CQ overrun support
 */
#if HAVE_DECL_IBV_EXP_CQ_IGNORE_OVERRUN
static inline int ibv_exp_cq_ignore_overrun(struct ibv_cq *cq)
{
    struct ibv_exp_cq_attr cq_attr = {0};
    cq_attr.comp_mask    = IBV_EXP_CQ_ATTR_CQ_CAP_FLAGS;
    cq_attr.cq_cap_flags = IBV_EXP_CQ_IGNORE_OVERRUN;
    return ibv_exp_modify_cq(cq, &cq_attr, IBV_EXP_CQ_CAP_FLAGS);
}
#else
static inline int ibv_exp_cq_ignore_overrun(struct ibv_cq *cq)
{
    errno = ENOSYS;
    return -1;
}
#endif /* HAVE_IBV_EXP_CQ_IGNORE_OVERRUN */


/*
 * Atomics support
 */
#if HAVE_DECL_IBV_EXP_ATOMIC_HCA_REPLY_BE
#  define IBV_EXP_HAVE_ATOMIC_HCA(_attr)            ((_attr)->exp_atomic_cap == IBV_EXP_ATOMIC_HCA)
#  define IBV_EXP_HAVE_ATOMIC_GLOB(_attr)           ((_attr)->exp_atomic_cap == IBV_EXP_ATOMIC_GLOB)
#  define IBV_EXP_HAVE_ATOMIC_HCA_REPLY_BE(_attr)   ((_attr)->exp_atomic_cap == IBV_EXP_ATOMIC_HCA_REPLY_BE)
#else
#  define IBV_EXP_HAVE_ATOMIC_HCA(_attr)            ((_attr)->atomic_cap == IBV_ATOMIC_HCA)
#  define IBV_EXP_HAVE_ATOMIC_GLOB(_attr)           ((_attr)->atomic_cap == IBV_ATOMIC_GLOB)
#  define IBV_EXP_HAVE_ATOMIC_HCA_REPLY_BE(_attr)   0
#endif /* HAVE_DECL_IBV_EXP_ATOMIC_HCA_REPLY_BE */


/* Ethernet link layer */
#if HAVE_DECL_IBV_LINK_LAYER_ETHERNET
#  define IBV_PORT_IS_LINK_LAYER_ETHERNET(_attr)    ((_attr)->link_layer == IBV_LINK_LAYER_ETHERNET)
#else
#  define IBV_PORT_IS_LINK_LAYER_ETHERNET(_attr)    0
#endif


/*
 * HW tag matching
 */
#if IBV_EXP_HW_TM
   /* TM (eager) is supported if tm_caps.max_num_tags is not 0. */
#  define IBV_DEVICE_TM_CAPS(_dev, _field)  ((_dev)->dev_attr.tm_caps._field)

   /* If the gap between SW and HW counters is more than 32K, all messages will
    * be dropped and RNR ACK will be returned. Set threshold to 16K to avoid
    * hitting this gap at all. */
#  define IBV_DEVICE_MAX_UNEXP_COUNT        UCS_BIT(14)

#  define IBV_DEVICE_MIN_UWQ_POST           33
#else
#  define IBV_DEVICE_TM_CAPS(_dev, _field)  0
#  define IBV_EXP_TM_CAP_RC                 0
#endif

#ifndef IBV_EXP_HW_TM_DC
#  define IBV_EXP_TM_CAP_DC                 0
#endif

#if !HAVE_DECL_IBV_EXP_CREATE_SRQ
#  define ibv_exp_create_srq_attr           ibv_srq_init_attr
#endif



typedef uint8_t uct_ib_uint24_t[3];

static inline void uct_ib_pack_uint24(uct_ib_uint24_t buf, const uint32_t qp_num)
{

    buf[0] = (qp_num >> 0)  & 0xFF;
    buf[1] = (qp_num >> 8)  & 0xFF;
    buf[2] = (qp_num >> 16) & 0xFF;
}

static inline uint32_t uct_ib_unpack_uint24(const uct_ib_uint24_t buf)
{
    return buf[0] | ((uint32_t)buf[1] << 8) | ((uint32_t)buf[2] << 16);
}

typedef struct uct_ib_qpnum {
    uct_ib_uint24_t qp_num;
} uct_ib_qpnum_t;

#endif /* UCT_IB_VERBS_H */
