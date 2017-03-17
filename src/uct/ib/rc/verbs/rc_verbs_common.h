/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_RC_VERBS_COMMON_H
#define UCT_RC_VERBS_COMMON_H

#include <ucs/arch/bitops.h>

#include <uct/ib/rc/base/rc_iface.h>
#include <uct/ib/rc/base/rc_ep.h>


/* definitions common to rc_verbs and dc_verbs go here */

typedef struct uct_rc_verbs_txcnt {
    uint16_t       pi;      /* producer (post_send) count */
    uint16_t       ci;      /* consumer (ibv_poll_cq) completion count */
} uct_rc_verbs_txcnt_t;


/**
 * RC/DC verbs interface configuration
 */
typedef struct uct_rc_verbs_iface_common_config {
    size_t                 max_am_hdr;
    unsigned               tx_max_wr;
    /* TODO flags for exp APIs */
} uct_rc_verbs_iface_common_config_t;


typedef struct uct_rc_verbs_iface_common {
    struct ibv_sge         inl_sge[2];
    void                   *am_inl_hdr;
    ucs_mpool_t            short_desc_mp;


    /* TODO: make a separate datatype */
    struct {
        size_t             notag_hdr_size;
        size_t             short_desc_size;
        size_t             max_inline;
    } config;
} uct_rc_verbs_iface_common_t;

extern ucs_config_field_t uct_rc_verbs_iface_common_config_table[];

void uct_rc_verbs_txcnt_init(uct_rc_verbs_txcnt_t *txcnt);

static inline void
uct_rc_verbs_txqp_posted(uct_rc_txqp_t *txqp, uct_rc_verbs_txcnt_t *txcnt,
                         uct_rc_iface_t *iface, int signaled)
{
    txcnt->pi++;
    uct_rc_txqp_posted(txqp, iface, 1, signaled);
}

static inline void
uct_rc_verbs_txqp_completed(uct_rc_txqp_t *txqp, uct_rc_verbs_txcnt_t *txcnt, uint16_t count)
{
    txcnt->ci += count;
    uct_rc_txqp_available_add(txqp, count);
}

ucs_status_t uct_rc_verbs_iface_common_init(uct_rc_verbs_iface_common_t *iface,
                                            uct_rc_iface_t *rc_iface,
                                            uct_rc_verbs_iface_common_config_t *config,
                                            uct_rc_iface_config_t *rc_config,
                                            size_t am_hdr_size);

void uct_rc_verbs_iface_common_cleanup(uct_rc_verbs_iface_common_t *iface);

ucs_status_t uct_rc_verbs_iface_prepost_recvs_common(uct_rc_iface_t *iface,
                                                     uct_rc_srq_t *srq);

void uct_rc_verbs_iface_common_query(uct_rc_verbs_iface_common_t *verbs_iface,
                                     uct_rc_iface_t *rc_iface, uct_iface_attr_t *iface_attr);

unsigned uct_rc_verbs_iface_post_recv_always(uct_rc_iface_t *iface,
                                             uct_rc_srq_t *srq, unsigned max);

static inline unsigned uct_rc_verbs_iface_post_recv_common(uct_rc_iface_t *iface,
                                                           uct_rc_srq_t *srq,
                                                           int fill)
{
    unsigned batch = iface->super.config.rx_max_batch;
    unsigned count;

    if (srq->available < batch) {
        if (ucs_likely(fill == 0)) {
            return 0;
        } else {
            count = srq->available;
        }
    } else {
        count = batch;
    }
    return uct_rc_verbs_iface_post_recv_always(iface, srq, count);
}

#define UCT_RC_VERBS_IFACE_FOREACH_TXWQE(_iface, _i, _wc, _num_wcs) \
      status = uct_ib_poll_cq((_iface)->super.send_cq, &_num_wcs, _wc); \
      if (status != UCS_OK) { \
          return; \
      } \
      UCS_STATS_UPDATE_COUNTER((_iface)->stats, UCT_RC_IFACE_STAT_TX_COMPLETION, _num_wcs); \
      for (_i = 0; _i < _num_wcs; ++_i)


/* TODO: think of a better name */
static inline int uct_rc_verbs_txcq_get_comp_count(struct ibv_wc *wc)
{
    return wc->wr_id + 1;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_verbs_iface_handle_am(uct_rc_iface_t *iface, struct ibv_wc *wc,
                             uct_rc_hdr_t *hdr, uint32_t length)
{
    uct_ib_iface_recv_desc_t *desc;
    uct_rc_iface_ops_t *rc_ops;
    void *udesc;
    ucs_status_t status;

    uct_ib_log_recv_completion(&iface->super, IBV_QPT_RC, wc, hdr, length,
                               uct_rc_ep_am_packet_dump);
    desc = (uct_ib_iface_recv_desc_t *)wc->wr_id;
    if (ucs_unlikely(hdr->am_id & UCT_RC_EP_FC_MASK)) {
        udesc = (char*)desc + iface->super.config.rx_headroom_offset;
        rc_ops = ucs_derived_of(iface->super.ops, uct_rc_iface_ops_t);
        status = rc_ops->fc_handler(iface, wc->qp_num, hdr,
                                    length - sizeof(*hdr),
                                    wc->imm_data, wc->slid, udesc);
        if (status == UCS_OK) {
            ucs_mpool_put_inline(desc);
        } else {
            uct_recv_desc_iface(udesc) = &iface->super.super.super;
        }
    } else {
        uct_ib_iface_invoke_am(&iface->super, hdr->am_id, hdr + 1,
                               length - sizeof(*hdr), desc);
    }
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_verbs_iface_poll_rx_common(uct_rc_iface_t *iface)
{
    uct_rc_hdr_t *hdr;
    unsigned i;
    ucs_status_t status;
    unsigned num_wcs = iface->super.config.rx_max_poll;
    struct ibv_wc wc[num_wcs];

    status = uct_ib_poll_cq(iface->super.recv_cq, &num_wcs, wc);
    if (status != UCS_OK) {
        goto out;
    }

    UCT_IB_IFACE_VERBS_FOREACH_RXWQE(&iface->super, i, hdr, wc, num_wcs) {
        uct_rc_verbs_iface_handle_am(iface, &wc[i], hdr, wc[i].byte_len);
    }
    iface->rx.srq.available += num_wcs;
    UCS_STATS_UPDATE_COUNTER(iface->stats, UCT_RC_IFACE_STAT_RX_COMPLETION, num_wcs);

out:
    uct_rc_verbs_iface_post_recv_common(iface, &iface->rx.srq, 0);
    return status;
}

static inline void
uct_rc_verbs_iface_fill_inl_am_sge(uct_rc_verbs_iface_common_t *iface,
                                   uint8_t id, uint64_t hdr,
                                   const void *buffer, unsigned length)
{
    uct_rc_am_short_hdr_t *am = (uct_rc_am_short_hdr_t*)((char*)iface->am_inl_hdr +
                                iface->config.notag_hdr_size);
    am->rc_hdr.am_id = id;
    am->am_hdr       = hdr;

    iface->inl_sge[0].addr      = (uintptr_t)iface->am_inl_hdr;
    iface->inl_sge[0].length    = sizeof(*am) + iface->config.notag_hdr_size;
    iface->inl_sge[1].addr      = (uintptr_t)buffer;
    iface->inl_sge[1].length    = length;
}

#define UCT_RC_VERBS_FILL_SGE(_wr, _sge, _length) \
    _wr.sg_list = &_sge; \
    _wr.num_sge = 1; \
    _sge.length = _length;

#define UCT_RC_VERBS_FILL_INL_PUT_WR(_iface, _raddr, _rkey, _buf, _len) \
    _iface->inl_rwrite_wr.wr.rdma.remote_addr = _raddr; \
    _iface->inl_rwrite_wr.wr.rdma.rkey        = uct_ib_md_direct_rkey(_rkey); \
    _iface->verbs_common.inl_sge[0].addr      = (uintptr_t)_buf; \
    _iface->verbs_common.inl_sge[0].length    = _len;

#define UCT_RC_VERBS_FILL_AM_BCOPY_WR(_wr, _sge, _length, _wr_opcode) \
    UCT_RC_VERBS_FILL_SGE(_wr, _sge, _length) \
    _wr_opcode = IBV_WR_SEND;

#define UCT_RC_VERBS_FILL_AM_ZCOPY_WR_IOV(_wr, _sge, _iovlen, _wr_opcode) \
    _wr.sg_list = _sge; \
    _wr.num_sge = _iovlen; \
    _wr_opcode  = IBV_WR_SEND;

#define UCT_RC_VERBS_FILL_RDMA_WR(_wr, _wr_opcode, _opcode, \
                                  _sge, _length, _raddr, _rkey) \
    UCT_RC_VERBS_FILL_SGE(_wr, _sge, _length) \
    _wr.wr.rdma.remote_addr = _raddr; \
    _wr.wr.rdma.rkey        = uct_ib_md_direct_rkey(_rkey); \
    _wr_opcode              = _opcode; \

#define UCT_RC_VERBS_FILL_RDMA_WR_IOV(_wr, _wr_opcode, _opcode, _sge, _sgelen, \
                                      _raddr, _rkey) \
    _wr.wr.rdma.remote_addr = _raddr; \
    _wr.wr.rdma.rkey        = uct_ib_md_direct_rkey(_rkey); \
    _wr.sg_list             = _sge; \
    _wr.num_sge             = _sgelen; \
    _wr_opcode              = _opcode;

#define UCT_RC_VERBS_FILL_DESC_WR(_wr, _desc) \
    { \
        struct ibv_sge *sge; \
        (_wr)->next    = NULL; \
        sge            = (_wr)->sg_list; \
        sge->addr      = (uintptr_t)(desc + 1); \
        sge->lkey      = (_desc)->lkey; \
    }

#define UCT_RC_VERBS_FILL_ATOMIC_WR(_wr, _wr_opcode, _sge, _opcode, \
                                    _compare_add, _swap, _remote_addr, _rkey) \
    UCT_RC_VERBS_FILL_SGE(_wr, _sge, sizeof(uint64_t)) \
    _wr_opcode                = _opcode; \
    _wr.wr.atomic.compare_add = _compare_add; \
    _wr.wr.atomic.swap        = _swap; \
    _wr.wr.atomic.remote_addr = _remote_addr; \
    _wr.wr.atomic.rkey        = _rkey;  \


#if HAVE_IB_EXT_ATOMICS
static inline void
uct_rc_verbs_fill_ext_atomic_wr(struct ibv_exp_send_wr *wr, struct ibv_sge *sge,
                                int opcode, uint32_t length, uint32_t compare_mask,
                                uint64_t compare_add, uint64_t swap, uint64_t remote_addr,
                                uct_rkey_t rkey, size_t atomic_mr_offset)
{
    sge->length        = length;
    wr->sg_list        = sge;
    wr->num_sge        = 1;
    wr->exp_opcode     = (enum ibv_exp_wr_opcode)opcode;
    wr->comp_mask      = 0;

    wr->ext_op.masked_atomics.log_arg_sz  = ucs_ilog2(length);
    wr->ext_op.masked_atomics.rkey        = uct_ib_resolve_atomic_rkey(rkey,
                                                                       atomic_mr_offset,
                                                                       &remote_addr);
    wr->ext_op.masked_atomics.remote_addr = remote_addr;

    switch (opcode) {
    case IBV_EXP_WR_EXT_MASKED_ATOMIC_CMP_AND_SWP:
        wr->ext_op.masked_atomics.wr_data.inline_data.op.cmp_swap.compare_mask = compare_mask;
        wr->ext_op.masked_atomics.wr_data.inline_data.op.cmp_swap.compare_val  = compare_add;
        wr->ext_op.masked_atomics.wr_data.inline_data.op.cmp_swap.swap_mask    = (uint64_t)(-1);
        wr->ext_op.masked_atomics.wr_data.inline_data.op.cmp_swap.swap_val     = swap;
        break;
    case IBV_EXP_WR_EXT_MASKED_ATOMIC_FETCH_AND_ADD:
        wr->ext_op.masked_atomics.wr_data.inline_data.op.fetch_add.add_val        = compare_add;
        wr->ext_op.masked_atomics.wr_data.inline_data.op.fetch_add.field_boundary = 0;
        break;
    }
}
#endif


#endif
